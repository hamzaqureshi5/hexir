//===- interpreter.c - Replay a host command list -------------------------===//
//
// Walks the PROGRAM section and executes it against the HAL.
//
// Everything read here comes off disk, so every index is bounds checked before
// use: a slot number, an executable index and a rodata offset are all just
// numbers in a file until they are validated.
//
//===----------------------------------------------------------------------===//

#include "hexir_runtime/vm.h"

#include "hexir_runtime/program.h"
#include "../kernels/reference_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEXIR_MAX_SLOTS 256

const char *hexir_kernel_kind_string(uint32_t kind) {
  switch (kind) {
  case HEXIR_KERNEL_MATMUL: return "matmul";
  case HEXIR_KERNEL_ADD: return "add";
  case HEXIR_KERNEL_RELU: return "relu";
  }
  return "unknown";
}

typedef struct {
  hexir_device_t *device;
  hexir_buffer_t *slots[HEXIR_MAX_SLOTS];
  const uint8_t *rodata;
  uint64_t rodata_size;
  const hexir_executable_entry_t *executables;
  uint32_t executable_count;
} vm_state_t;

static void vm_release(vm_state_t *vm) {
  for (int i = 0; i < HEXIR_MAX_SLOTS; ++i)
    if (vm->slots[i])
      hexir_buffer_release(vm->slots[i]);
}

static hexir_status_t vm_bind(vm_state_t *vm, uint64_t slot, size_t size,
                              hexir_buffer_t **out) {
  if (slot >= HEXIR_MAX_SLOTS)
    return HEXIR_ERROR_INVALID_MODULE;
  if (vm->slots[slot]) {
    hexir_buffer_release(vm->slots[slot]);
    vm->slots[slot] = NULL;
  }
  hexir_status_t status = hexir_buffer_allocate(
      vm->device, size, HEXIR_MEMORY_DEVICE_LOCAL, &vm->slots[slot]);
  if (status != HEXIR_OK)
    return status;
  *out = vm->slots[slot];
  return HEXIR_OK;
}

static hexir_buffer_t *vm_slot(vm_state_t *vm, uint64_t slot) {
  if (slot >= HEXIR_MAX_SLOTS)
    return NULL;
  return vm->slots[slot];
}

static hexir_status_t vm_dispatch(vm_state_t *vm, const uint64_t *operands,
                                  uint32_t count) {
  if (count < 2)
    return HEXIR_ERROR_INVALID_MODULE;
  uint64_t index = operands[0];
  uint64_t arg_count = operands[1];
  if (index >= vm->executable_count || count != 2 + arg_count)
    return HEXIR_ERROR_INVALID_MODULE;

  const hexir_executable_entry_t *exe = &vm->executables[index];

  /* Destination is the last argument: prim funcs are destination-passing. */
  double *args[4];
  if (arg_count > 4)
    return HEXIR_ERROR_INVALID_MODULE;
  for (uint64_t i = 0; i < arg_count; ++i) {
    hexir_buffer_t *buffer = vm_slot(vm, operands[2 + i]);
    if (!buffer)
      return HEXIR_ERROR_INVALID_MODULE;
    args[i] = (double *)hexir_buffer_host_pointer(buffer);
    if (!args[i])
      return HEXIR_ERROR_DEVICE;
  }

  size_t elems = (size_t)exe->m * exe->n;
  switch (exe->kind) {
  case HEXIR_KERNEL_MATMUL:
    if (arg_count != 3)
      return HEXIR_ERROR_INVALID_MODULE;
    hexir_ref_matmul_f64(args[0], args[1], args[2], exe->m, exe->n, exe->k);
    break;
  case HEXIR_KERNEL_ADD:
    if (arg_count != 3)
      return HEXIR_ERROR_INVALID_MODULE;
    hexir_ref_add_f64(args[0], args[1], args[2], elems);
    break;
  case HEXIR_KERNEL_RELU:
    if (arg_count != 2)
      return HEXIR_ERROR_INVALID_MODULE;
    hexir_ref_relu_f64(args[0], args[1], elems);
    break;
  default:
    return HEXIR_ERROR_UNIMPLEMENTED;
  }
  return hexir_device_wait(vm->device);
}

/* Matches the printf lowering in the compiler, so JIT and runtime output are
 * byte-for-byte comparable. */
static hexir_status_t vm_print(vm_state_t *vm, uint64_t slot, uint64_t rows,
                               uint64_t cols) {
  hexir_buffer_t *buffer = vm_slot(vm, slot);
  if (!buffer)
    return HEXIR_ERROR_INVALID_MODULE;
  if (rows * cols * sizeof(double) > hexir_buffer_size(buffer))
    return HEXIR_ERROR_INVALID_MODULE;

  const double *data = (const double *)hexir_buffer_host_pointer(buffer);
  if (!data)
    return HEXIR_ERROR_DEVICE;
  for (uint64_t i = 0; i < rows; ++i) {
    for (uint64_t j = 0; j < cols; ++j)
      printf("%f ", data[i * cols + j]);
    printf("\n");
  }
  return HEXIR_OK;
}

hexir_status_t hexir_execute(const hexir_module_t *module,
                             hexir_device_t *device, const char *entry) {
  if (!module || !device || !entry)
    return HEXIR_ERROR_INVALID_ARGUMENT;

  const void *symbols_data = NULL, *program_data = NULL;
  uint64_t symbols_size = 0, program_size = 0;
  if (hexir_module_section(module, HEXIR_SECTION_SYMBOLS, &symbols_data,
                           &symbols_size) != HEXIR_OK ||
      hexir_module_section(module, HEXIR_SECTION_PROGRAM, &program_data,
                           &program_size) != HEXIR_OK)
    return HEXIR_ERROR_INVALID_MODULE;

  vm_state_t vm;
  memset(&vm, 0, sizeof(vm));
  vm.device = device;

  const void *data = NULL;
  uint64_t size = 0;
  if (hexir_module_section(module, HEXIR_SECTION_RODATA, &data, &size) ==
      HEXIR_OK) {
    vm.rodata = (const uint8_t *)data;
    vm.rodata_size = size;
  }
  if (hexir_module_section(module, HEXIR_SECTION_EXECUTABLES, &data, &size) ==
      HEXIR_OK) {
    vm.executables = (const hexir_executable_entry_t *)data;
    vm.executable_count = (uint32_t)(size / sizeof(hexir_executable_entry_t));
  }

  const hexir_symbol_entry_t *symbols =
      (const hexir_symbol_entry_t *)symbols_data;
  uint64_t symbol_count = symbols_size / sizeof(hexir_symbol_entry_t);
  uint64_t pc = 0;
  int found = 0;
  for (uint64_t i = 0; i < symbol_count; ++i) {
    if (strncmp(symbols[i].name, entry, HEXIR_NAME_SIZE) == 0) {
      pc = symbols[i].program_offset;
      found = 1;
      break;
    }
  }
  if (!found)
    return HEXIR_ERROR_NOT_FOUND;

  const uint8_t *program = (const uint8_t *)program_data;
  hexir_status_t status = HEXIR_OK;

  while (status == HEXIR_OK) {
    if (pc + sizeof(hexir_command_header_t) > program_size) {
      status = HEXIR_ERROR_INVALID_MODULE;
      break;
    }
    hexir_command_header_t header;
    memcpy(&header, program + pc, sizeof(header));
    pc += sizeof(header);

    uint64_t operand_bytes = (uint64_t)header.operand_count * sizeof(uint64_t);
    if (pc + operand_bytes > program_size) {
      status = HEXIR_ERROR_INVALID_MODULE;
      break;
    }
    const uint64_t *operands = (const uint64_t *)(const void *)(program + pc);
    pc += operand_bytes;

    switch (header.kind) {
    case HEXIR_CMD_END:
      vm_release(&vm);
      return HEXIR_OK;

    case HEXIR_CMD_ALLOC: {
      hexir_buffer_t *buffer = NULL;
      if (header.operand_count != 2)
        status = HEXIR_ERROR_INVALID_MODULE;
      else
        status = vm_bind(&vm, operands[0], (size_t)operands[1], &buffer);
      break;
    }

    case HEXIR_CMD_CONST: {
      if (header.operand_count != 3) {
        status = HEXIR_ERROR_INVALID_MODULE;
        break;
      }
      uint64_t offset = operands[1], bytes = operands[2];
      if (offset > vm.rodata_size || bytes > vm.rodata_size - offset) {
        status = HEXIR_ERROR_INVALID_MODULE;
        break;
      }
      hexir_buffer_t *buffer = NULL;
      status = vm_bind(&vm, operands[0], (size_t)bytes, &buffer);
      if (status == HEXIR_OK)
        status = hexir_buffer_write(buffer, vm.rodata + offset, (size_t)bytes);
      break;
    }

    case HEXIR_CMD_DISPATCH:
      status = vm_dispatch(&vm, operands, header.operand_count);
      break;

    case HEXIR_CMD_PRINT:
      if (header.operand_count != 3)
        status = HEXIR_ERROR_INVALID_MODULE;
      else
        status = vm_print(&vm, operands[0], operands[1], operands[2]);
      break;

    default:
      status = HEXIR_ERROR_INVALID_MODULE;
      break;
    }
  }

  vm_release(&vm);
  return status;
}
