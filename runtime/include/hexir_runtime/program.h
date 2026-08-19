//===- program.h - Host program encoding -------------------------*- C -*-===//
//
// The encoding of the PROGRAM and EXECUTABLES sections of a module.
//
// This header is the contract between the two halves of the project, and it
// lives on the runtime side on purpose: the compiler emits the format the
// runtime defines, never the other way round. The compiler includes this
// header; nothing links back.
//
// The host program is a flat command list, not bytecode. Everything the
// compiler can currently produce is straight-line dataflow with no control
// flow, so a list of commands to replay is sufficient and an interpreter with
// branches would be unused machinery. Bytecode becomes necessary when
// dynamic shapes or control flow arrive.
//
// Buffers are referred to by slot: a small dense index space the compiler
// assigns, which the runtime resolves to real allocations.
//
//===----------------------------------------------------------------------===//
#ifndef HEXIR_RUNTIME_PROGRAM_H
#define HEXIR_RUNTIME_PROGRAM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  HEXIR_CMD_END = 0,
  /* slot, byte_size -- a zeroed destination buffer */
  HEXIR_CMD_ALLOC = 1,
  /* slot, rodata_offset, byte_size -- bind constant data from RODATA */
  HEXIR_CMD_CONST = 2,
  /* executable_index, arg_count, slot... -- last slot is the destination */
  HEXIR_CMD_DISPATCH = 3,
  /* slot, rows, cols */
  HEXIR_CMD_PRINT = 4,
} hexir_command_kind_t;

/* Followed by `operand_count` little-endian uint64_t operands. */
typedef struct {
  uint32_t kind;
  uint32_t operand_count;
} hexir_command_header_t;

/* What a dispatch actually computes.
 *
 * These name a computation rather than carrying machine code: the EXECUTABLES
 * section holds kernel *descriptors*, and the runtime supplies the
 * implementation. That is a bootstrapping step -- it makes the artifact, the
 * loader, the HAL and the command list real and testable end to end. Replacing
 * a descriptor with an embedded CUBIN (from gpu.binary) or a host object is
 * the next step and does not change the container. */
typedef enum {
  HEXIR_KERNEL_MATMUL = 1,
  HEXIR_KERNEL_ADD = 2,
  HEXIR_KERNEL_RELU = 3,
} hexir_kernel_kind_t;

#define HEXIR_NAME_SIZE 32

typedef struct {
  char name[HEXIR_NAME_SIZE];
  uint32_t kind;      /* hexir_kernel_kind_t */
  uint32_t device;    /* hexir_device_kind_t: where the compiler placed it */
  uint32_t m, n, k;   /* extents; k is 0 for element-wise kernels */
  uint32_t elem_size; /* bytes per element */
  uint32_t reserved;
} hexir_executable_entry_t;

typedef struct {
  char name[HEXIR_NAME_SIZE];
  uint32_t program_offset; /* byte offset into PROGRAM */
  uint32_t reserved;
} hexir_symbol_entry_t;

const char *hexir_kernel_kind_string(uint32_t kind);

#ifdef __cplusplus
}
#endif
#endif // HEXIR_RUNTIME_PROGRAM_H
