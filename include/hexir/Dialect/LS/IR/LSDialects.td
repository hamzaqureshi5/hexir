//===- LSDialects.td - Local-system CPU/GPU dialects -------*- tablegen -*-===//
//
// Lightweight target dialects used to show heterogeneous partitioning.
//
//===----------------------------------------------------------------------===//

#ifndef LS_DIALECTS
#define LS_DIALECTS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LSCPU_Dialect : Dialect {
  let name = "ls_cpu";
  let cppNamespace = "::mlir::ls_cpu";
  let summary = "local-system CPU model dialect";
  let description = [{
    The ls_cpu dialect models operations selected for CPU execution.
  }];
}

def LSGPU_Dialect : Dialect {
  let name = "ls_gpu";
  let cppNamespace = "::mlir::ls_gpu";
  let summary = "local-system GPU model dialect";
  let description = [{
    The ls_gpu dialect models operations selected for GPU execution.
  }];
}

class LSCPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<LSCPU_Dialect, mnemonic, traits>;

class LSGPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<LSGPU_Dialect, mnemonic, traits>;

def LSCPU_AddOp : LSCPU_Op<"add", [Pure]> {
  let summary = "CPU add model operation";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor:$result);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def LSCPU_MulOp : LSCPU_Op<"mul", [Pure]> {
  let summary = "CPU multiply model operation";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor:$result);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def LSCPU_MatmulOp : LSCPU_Op<"matmul", [Pure]> {
  let summary = "CPU matrix multiply model operation";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor:$result);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def LSCPU_ReluOp : LSCPU_Op<"relu", [Pure]> {
  let summary = "CPU ReLU model operation";
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$result);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];
}

def LSGPU_AddOp : LSGPU_Op<"add", [Pure]> {
  let summary = "GPU add model operation";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor:$result);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def LSGPU_MulOp : LSGPU_Op<"mul", [Pure]> {
  let summary = "GPU multiply model operation";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor:$result);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def LSGPU_MatmulOp : LSGPU_Op<"matmul", [Pure]> {
  let summary = "GPU matrix multiply model operation";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor:$result);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def LSGPU_ReluOp : LSGPU_Op<"relu", [Pure]> {
  let summary = "GPU ReLU model operation";
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$result);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];
}

#endif // LS_DIALECTS
