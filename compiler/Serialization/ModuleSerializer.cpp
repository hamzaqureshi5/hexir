//===- ModuleSerializer.cpp - Emit a loadable module ----------------------===//
//
// Walks the kernel-level IR and writes the four sections the runtime expects.
//
//   hexir.constant  -> bytes in RODATA + a CONST command binding a slot
//   hexir.call_tir  -> an ALLOC for the result + a DISPATCH
//   hexir.print     -> a PRINT command
//   hextir.prim_func-> an entry in EXECUTABLES
//
// Slots are a dense index space assigned here: each SSA value that survives
// into the program gets one, and the runtime resolves it to an allocation.
//
//===----------------------------------------------------------------------===//

#include "hexir/Serialization/ModuleSerializer.h"

#include "hexir/Dialect/HexTIR/IR/HexTIRDialect.h"
#include "hexir/Dialect/Hexir/IR/HexirDialect.h"

#include "hexir_runtime/module.h"
#include "hexir_runtime/program.h"

#include "hexir/Conversion/Passes.h"

#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstring>
#include <string>

using namespace mlir;

namespace {

/// Little-endian byte buffer. The format is fixed-endian so a module written
/// on one machine loads on another.
struct ByteBuffer {
  llvm::SmallVector<char> bytes;

  void u32(uint32_t v) { raw(&v, sizeof(v)); }
  void u64(uint64_t v) { raw(&v, sizeof(v)); }
  void raw(const void *data, size_t size) {
    const char *p = static_cast<const char *>(data);
    bytes.append(p, p + size);
  }
  void padTo(size_t alignment) {
    while (bytes.size() % alignment)
      bytes.push_back(0);
  }
  size_t size() const { return bytes.size(); }
};

/// Fixed-width name field; longer names are truncated rather than rejected,
/// since a truncated descriptor is still loadable and the name is diagnostic.
void writeName(char (&dst)[HEXIR_NAME_SIZE], llvm::StringRef name) {
  std::memset(dst, 0, HEXIR_NAME_SIZE);
  std::memcpy(dst, name.data(), std::min<size_t>(name.size(), HEXIR_NAME_SIZE - 1));
}

uint32_t kernelKindFromAttr(llvm::StringRef kernel) {
  if (kernel == "matmul")
    return HEXIR_KERNEL_MATMUL;
  if (kernel == "add")
    return HEXIR_KERNEL_ADD;
  return HEXIR_KERNEL_RELU;
}

struct Serializer {
  ModuleOp module;
  std::string gpuChip;

  ByteBuffer rodata, program, executables, images, symbols;
  llvm::DenseMap<Value, uint64_t> slotOf;
  llvm::DenseMap<llvm::StringRef, uint32_t> executableIndex;
  uint64_t nextSlot = 0;

  Serializer(ModuleOp module, StringRef gpuChip)
      : module(module), gpuChip(gpuChip.str()) {}

  uint64_t assignSlot(Value v) {
    uint64_t slot = nextSlot++;
    slotOf[v] = slot;
    return slot;
  }

  void command(uint32_t kind, llvm::ArrayRef<uint64_t> operands) {
    program.u32(kind);
    program.u32(static_cast<uint32_t>(operands.size()));
    for (uint64_t operand : operands)
      program.u64(operand);
  }

  /// Byte size of a ranked tensor of a fixed-width element type.
  static uint64_t byteSize(RankedTensorType type) {
    uint64_t elems = 1;
    for (int64_t dim : type.getShape())
      elems *= static_cast<uint64_t>(dim);
    return elems * (type.getElementTypeBitWidth() / 8);
  }

  /// Compile every cuda-placed kernel to a device image.
  ///
  /// Done on a clone: the module being serialized has to stay at the kernel
  /// level, because that is what the command list is built from. The clone is
  /// lowered hextir -> gpu -> NVVM -> CUBIN and only the blobs come back.
  ///
  /// Bare-pointer calling convention on purpose: without it every memref
  /// argument expands into seven scalars and the runtime would have to build
  /// MLIR's descriptor layout by hand. With it a kernel argument is one device
  /// pointer.
  llvm::StringMap<std::string> compileDeviceImages(
      llvm::StringMap<std::pair<uint32_t, uint32_t>> &geometry) {
    llvm::StringMap<std::string> blobs;

    bool anyCuda = false;
    for (auto fn : module.getOps<hextir::PrimFuncOp>())
      if (auto device = fn->getAttrOfType<StringAttr>("device"))
        anyCuda |= device.getValue() == "cuda";
    if (!anyCuda)
      return blobs;

    OwningOpRef<ModuleOp> clone(module.clone());

    // Two runs on purpose. gpu-module-to-binary erases the gpu.module it
    // compiles, so the launch geometry has to be read off it in between.
    {
      PassManager pm(module.getContext());
      pm.addPass(mlir::hexir::createHexTIRToGPUPass());
      // The kernel still has scf.for for any unbound axis (the reduction of a
      // matmul), and it has to be flat before it can become LLVM IR.
      pm.addPass(createSCFToControlFlowPass());
      if (failed(pm.run(*clone))) {
        module.emitWarning("could not lower cuda kernels to the gpu dialect; "
                           "the module will carry descriptors only");
        return blobs;
      }
    }

    clone->walk([&](gpu::GPUModuleOp gpuModule) {
      auto grid = gpuModule->getAttrOfType<IntegerAttr>("hexir.grid_x");
      auto block = gpuModule->getAttrOfType<IntegerAttr>("hexir.block_x");
      StringRef name = gpuModule.getName();
      name.consume_back("_module");
      geometry[name] = {grid ? static_cast<uint32_t>(grid.getInt()) : 1u,
                        block ? static_cast<uint32_t>(block.getInt()) : 1u};
    });

    {
      PassManager pm(module.getContext());
      GpuNVVMAttachTargetOptions nvvmOpts;
      nvvmOpts.chip = gpuChip;
      nvvmOpts.features = "+ptx80";
      nvvmOpts.optLevel = 3;
      pm.addPass(createGpuNVVMAttachTarget(nvvmOpts));

      // Bare-pointer calling convention on purpose: without it every memref
      // argument expands into seven scalars and the runtime would have to
      // build MLIR's descriptor layout by hand. With it a kernel argument is
      // one device pointer.
      ConvertGpuOpsToNVVMOpsOptions nvvmConv;
      nvvmConv.useBarePtrCallConv = true;
      pm.nest<gpu::GPUModuleOp>().addPass(
          createConvertGpuOpsToNVVMOps(nvvmConv));
      // A partial conversion: without this the leftover casts make the
      // translation to LLVM IR fail.
      pm.nest<gpu::GPUModuleOp>().addPass(createReconcileUnrealizedCastsPass());
      pm.addPass(createGpuModuleToBinaryPass());

      if (failed(pm.run(*clone))) {
        module.emitWarning("could not compile cuda kernels to device code; the "
                           "module will carry descriptors only");
        return blobs;
      }
    }

    clone->walk([&](gpu::BinaryOp binary) {
      if (binary.getObjects().empty())
        return;
      auto object = dyn_cast<gpu::ObjectAttr>(binary.getObjects()[0]);
      if (!object)
        return;
      // gpu.binary is named after the gpu.module, which hextir-to-gpu named
      // "<kernel>_module".
      StringRef name = binary.getName();
      name.consume_back("_module");
      blobs[name] = object.getObject().getValue().str();
    });
    return blobs;
  }

  LogicalResult collectExecutables() {
    llvm::StringMap<std::pair<uint32_t, uint32_t>> geometry;
    llvm::StringMap<std::string> blobs = compileDeviceImages(geometry);

    SmallVector<hexir_executable_entry_t> entries;
    for (auto fn : module.getOps<hextir::PrimFuncOp>()) {
      auto kernel = fn->getAttrOfType<StringAttr>("hexir.kernel");
      if (!kernel)
        return fn.emitError("prim func has no hexir.kernel attribute; it was "
                            "not produced by hexir-lower-to-tir");

      auto argTypes = fn.getArgumentTypes();
      auto dstTy = dyn_cast<MemRefType>(argTypes.back());
      if (!dstTy || dstTy.getRank() != 2)
        return fn.emitError("expected a rank-2 memref destination");

      hexir_executable_entry_t entry;
      std::memset(&entry, 0, sizeof(entry));
      writeName(entry.name, fn.getSymName());
      entry.kind = kernelKindFromAttr(kernel.getValue());
      auto device = fn->getAttrOfType<StringAttr>("device");
      entry.device = (device && device.getValue() == "cuda") ? 1u : 0u;
      entry.m = static_cast<uint32_t>(dstTy.getShape()[0]);
      entry.n = static_cast<uint32_t>(dstTy.getShape()[1]);
      if (entry.kind == HEXIR_KERNEL_MATMUL) {
        auto lhsTy = cast<MemRefType>(argTypes[0]);
        entry.k = static_cast<uint32_t>(lhsTy.getShape()[1]);
      }
      entry.elem_size =
          static_cast<uint32_t>(dstTy.getElementTypeBitWidth() / 8);

      auto blob = blobs.find(fn.getSymName());
      if (blob != blobs.end()) {
        images.padTo(8);
        entry.image_offset = static_cast<uint32_t>(images.size()); // fixed up
        entry.image_size = static_cast<uint32_t>(blob->second.size());
        images.raw(blob->second.data(), blob->second.size());
        auto dims = geometry.find(fn.getSymName());
        entry.grid_x = dims != geometry.end() ? dims->second.first : 1u;
        entry.block_x = dims != geometry.end() ? dims->second.second : 1u;
      }

      executableIndex[fn.getSymName()] =
          static_cast<uint32_t>(executableIndex.size());
      entries.push_back(entry);
    }

    // Images sit after the header and the entry table, so offsets are only
    // final once the table size is known.
    uint32_t base = static_cast<uint32_t>(sizeof(hexir_executable_header_t) +
                                          entries.size() * sizeof(entries[0]));
    base += (8 - (base % 8)) % 8;

    hexir_executable_header_t header;
    std::memset(&header, 0, sizeof(header));
    header.count = static_cast<uint32_t>(entries.size());
    executables.raw(&header, sizeof(header));
    for (hexir_executable_entry_t &entry : entries) {
      if (entry.image_size)
        entry.image_offset += base;
      executables.raw(&entry, sizeof(entry));
    }
    executables.padTo(8);
    executables.raw(images.bytes.data(), images.size());
    return success();
  }

  LogicalResult serializeFunc(func::FuncOp fn) {
    hexir_symbol_entry_t symbol;
    std::memset(&symbol, 0, sizeof(symbol));
    writeName(symbol.name, fn.getName());
    symbol.program_offset = static_cast<uint32_t>(program.size());
    symbols.raw(&symbol, sizeof(symbol));

    for (Operation &op : fn.getBody().front()) {
      if (auto constant = dyn_cast<mlir::hexir::ConstantOp>(op)) {
        // Constants with no users never reach the runtime.
        if (constant->use_empty())
          continue;
        auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
        if (!dense)
          return constant.emitError("expected a dense constant");
        auto type = cast<RankedTensorType>(constant.getType());

        uint64_t offset = rodata.size();
        // Raw data goes in verbatim: the runtime uses it from the mapping.
        llvm::ArrayRef<char> raw = dense.getRawData();
        rodata.raw(raw.data(), raw.size());
        rodata.padTo(8);

        uint64_t slot = assignSlot(constant.getResult());
        command(HEXIR_CMD_CONST, {slot, offset, byteSize(type)});
        continue;
      }

      if (auto call = dyn_cast<mlir::hexir::CallTIROp>(op)) {
        auto it = executableIndex.find(call.getCallee());
        if (it == executableIndex.end())
          return call.emitError("callee has no executable entry");

        auto resultTy = cast<RankedTensorType>(call.getResult().getType());
        uint64_t dst = assignSlot(call.getResult());
        command(HEXIR_CMD_ALLOC, {dst, byteSize(resultTy)});

        llvm::SmallVector<uint64_t> operands;
        operands.push_back(it->second);
        operands.push_back(call.getArgs().size() + 1);
        for (Value arg : call.getArgs()) {
          auto slot = slotOf.find(arg);
          if (slot == slotOf.end())
            return call.emitError("argument has no slot; it is not produced by "
                                  "a constant or a call in this function");
          operands.push_back(slot->second);
        }
        operands.push_back(dst);
        command(HEXIR_CMD_DISPATCH, operands);
        continue;
      }

      if (auto print = dyn_cast<mlir::hexir::PrintOp>(op)) {
        auto slot = slotOf.find(print.getInput());
        if (slot == slotOf.end())
          return print.emitError("operand has no slot");
        auto type = dyn_cast<RankedTensorType>(print.getInput().getType());
        if (!type || type.getRank() != 2)
          return print.emitError("expected a rank-2 tensor");
        command(HEXIR_CMD_PRINT,
                {slot->second, static_cast<uint64_t>(type.getShape()[0]),
                 static_cast<uint64_t>(type.getShape()[1])});
        continue;
      }

      if (isa<func::ReturnOp>(op))
        continue;

      return op.emitError("cannot serialize this op into a host program");
    }

    command(HEXIR_CMD_END, {});
    return success();
  }

  LogicalResult run(llvm::raw_ostream &os) {
    if (failed(collectExecutables()))
      return failure();

    for (auto fn : module.getOps<func::FuncOp>())
      if (failed(serializeFunc(fn)))
        return failure();

    // Header, section table, then 8-byte aligned payloads.
    struct Section {
      uint32_t kind;
      const ByteBuffer *data;
    };
    const Section sections[] = {
        {HEXIR_SECTION_SYMBOLS, &symbols},
        {HEXIR_SECTION_PROGRAM, &program},
        {HEXIR_SECTION_RODATA, &rodata},
        {HEXIR_SECTION_EXECUTABLES, &executables},
    };
    const uint32_t count = sizeof(sections) / sizeof(sections[0]);

    ByteBuffer out;
    hexir_module_header_t header;
    std::memset(&header, 0, sizeof(header));
    std::memcpy(header.magic, HEXIR_MODULE_MAGIC, HEXIR_MODULE_MAGIC_SIZE);
    header.version = HEXIR_MODULE_VERSION;
    header.section_count = count;
    out.raw(&header, sizeof(header));

    // Offsets are absolute, so compute them before writing the table.
    uint64_t offset = sizeof(hexir_module_header_t) +
                      count * sizeof(hexir_section_entry_t);
    hexir_section_entry_t entries[count];
    for (uint32_t i = 0; i < count; ++i) {
      offset += (8 - (offset % 8)) % 8;
      std::memset(&entries[i], 0, sizeof(entries[i]));
      entries[i].kind = sections[i].kind;
      entries[i].offset = offset;
      entries[i].size = sections[i].data->size();
      offset += sections[i].data->size();
    }
    out.raw(entries, sizeof(entries));

    for (uint32_t i = 0; i < count; ++i) {
      out.padTo(8);
      out.raw(sections[i].data->bytes.data(), sections[i].data->size());
    }

    os.write(out.bytes.data(), out.size());
    return success();
  }
};

} // namespace

LogicalResult mlir::hexir::serializeToHXB(ModuleOp module,
                                          llvm::raw_ostream &os,
                                          llvm::StringRef gpuChip) {
  return Serializer(module, gpuChip).run(os);
}
