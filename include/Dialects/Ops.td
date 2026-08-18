//===- Ops.td - Hexir dialect operation definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the Hexir dialect.
//
//===----------------------------------------------------------------------===//

#ifndef HEXIR_OPS
#define HEXIR_OPS

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ShapeInferenceInterface.td"

// Provide a definition of the 'hexir' dialect in the ODS framework so that we
// can define our operations.
def Hexir_Dialect : Dialect {
  let name = "hexir";
  let cppNamespace = "::mlir::hexir";

  // We set this bit to generate a declaration of the `materializeConstant`
  // method so that we can materialize constants for our hexir operations.
  //let hasConstantMaterializer = 1; Original is 1
  let hasConstantMaterializer = 0;

  // We set this bit to generate the declarations for the dialect's type parsing
  // and printing hooks.
  let useDefaultTypePrinterParser = 1;

}

// Base class for hexir dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Hexir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hexir_Dialect, mnemonic, traits>;

// Provide a definition for the Hexir StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef. We use `DialectType`
// to demarcate the StructType as belonging to the Hexir dialect.
def Hexir_StructType :
    DialectType<Hexir_Dialect, CPred<"::llvm::isa<StructType>($_self)">,
                "Hexir struct type">;

// Provide a definition of the types that are used within the Hexir dialect.
def Hexir_Type : AnyTypeOf<[F64Tensor, Hexir_StructType]>;

//===----------------------------------------------------------------------===//
// Hexir Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// We define a MLP operation by inheriting from our base 'Hexir_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'Pure' as it is a pure operation
// and may be removed if dead.
def ConstantOp : Hexir_Op<"constant",
    [Pure]> {//      ConstantLike,Pure 
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = hexir.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `ConstantOp::create(builder, ...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<(ins "double":$value)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;

  // Set the folder bit so that we can implement constant folders.
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : Hexir_Op<"add",
    [Pure]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}
//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

// def MulOp : Hexir_Op<"mul",
//     [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
//   let summary = "element-wise multiplication operation";
//   let description = [{
//     The "mul" operation performs element-wise multiplication between two
//     tensors. The shapes of the tensor operands are expected to match.
//   }];

//   let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
//   let results = (outs F64Tensor);

//   // Indicate that the operation has a custom parser and printer method.
//   let hasCustomAssemblyFormat = 1;

//   // Allow building a MulOp with from the two input operands.
//   let builders = [
//     OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
//   ];
// }

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

def ReluOp : Hexir_Op<"relu",
    [Pure]> {

  let summary = "element-wise ReLU operation";
  let description = [{
    The `relu` operation applies the Rectified Linear Unit function
    element-wise to the input tensor:

      relu(x) = max(0, x)

    The input and output tensor shapes are identical.
  }];

  let arguments = (ins F64Tensor:$input);
  let results   = (outs F64Tensor:$result);

  // Custom printer/parser (optional, but consistent with AddOp)
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];
}

// ===----------------------------------------------------------------------===//
// LeakyReLU
// ===----------------------------------------------------------------------===//
def LeakyReluOp : Hexir_Op<"leaky_relu", [Pure]> {
  let summary = "leaky rectified linear unit";
  let description = [{
    Applies max(x, alpha * x) element-wise.
  }];
  let arguments = (ins F64Tensor:$input);
  let results   = (outs F64Tensor:$result);
  let hasCustomAssemblyFormat = 1;
  let builders = [ OpBuilder<(ins "Value":$input)> ];


}

// ===----------------------------------------------------------------------===//
// ELU
// ===----------------------------------------------------------------------===//
def EluOp : Hexir_Op<"elu", [Pure]> {
  let summary = "exponential linear unit";
  let description = [{
    Applies x if x > 0, otherwise alpha * (exp(x) - 1).
  }];
  let arguments = (ins F64Tensor:$input);
  let results   = (outs F64Tensor:$result);
  let hasCustomAssemblyFormat = 1;
  let builders = [ OpBuilder<(ins "Value":$input)> ];

}

// ===----------------------------------------------------------------------===//
// Sigmoid
// ===----------------------------------------------------------------------===//
def SigmoidOp : Hexir_Op<"sigmoid", [Pure]> {
  let summary = "sigmoid activation";
  let description = [{
    Applies 1 / (1 + exp(-x)) element-wise.
  }];
  let arguments = (ins F64Tensor:$input);
  let results   = (outs F64Tensor:$result);
  let hasCustomAssemblyFormat = 1;
    let builders = [ OpBuilder<(ins "Value":$input)> ];

}

// ===----------------------------------------------------------------------===//
// Tanh
// ===----------------------------------------------------------------------===//
def TanhOp : Hexir_Op<"tanh", [Pure]> {
  let summary = "hyperbolic tangent activation";
  let description = [{
    Applies tanh(x) element-wise.
  }];
  let arguments = (ins F64Tensor:$input);
  let results   = (outs F64Tensor:$result);
  let hasCustomAssemblyFormat = 1;
    let builders = [ OpBuilder<(ins "Value":$input)> ];

}

// ===----------------------------------------------------------------------===//
// Softmax
// ===----------------------------------------------------------------------===//
def SoftmaxOp : Hexir_Op<"softmax", [Pure]> {
  let summary = "softmax normalization operation";
  let description = [{
    Normalizes values along the last dimension so that they sum to 1.
  }];
  let arguments = (ins F64Tensor:$input);
  let results   = (outs F64Tensor:$result);
  let hasCustomAssemblyFormat = 1;
    let builders = [ OpBuilder<(ins "Value":$input)> ];

}

// ===----------------------------------------------------------------------===//
// GELU
// ===----------------------------------------------------------------------===//
def GeluOp : Hexir_Op<"gelu", [Pure]> {
  let summary = "Gaussian Error Linear Unit";
  let description = [{
    Applies GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))).
  }];
  let arguments = (ins F64Tensor:$input);
  let results   = (outs F64Tensor:$result);
  let hasCustomAssemblyFormat = 1;
    let builders = [ OpBuilder<(ins "Value":$input)> ];

}

// ===----------------------------------------------------------------------===//
// Swish
// ===----------------------------------------------------------------------===//
def SwishOp : Hexir_Op<"swish", [Pure]> {
  let summary = "swish activation";
  let description = [{
    Applies x * sigmoid(x) element-wise.
  }];
  let arguments = (ins F64Tensor:$input);
  let results   = (outs F64Tensor:$result);
  let hasCustomAssemblyFormat = 1;
    let builders = [ OpBuilder<(ins "Value":$input)> ];

}

// ===----------------------------------------------------------------------===//
// Mish
// ===----------------------------------------------------------------------===//
def MishOp : Hexir_Op<"mish", [Pure]> {
  let summary = "mish activation";
  let description = [{
    Applies x * tanh(softplus(x)) element-wise.
  }];
  let arguments = (ins F64Tensor:$input);
  let results   = (outs F64Tensor:$result);
  let hasCustomAssemblyFormat = 1;
    let builders = [ OpBuilder<(ins "Value":$input)> ];

}

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//

def LinearOp : Hexir_Op<"linear",
    [Pure]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$input, F64Tensor:$weight);
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$weight)>
  ];
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

// def CastOp : Hexir_Op<"cast", [
//      DeclareOpInterfaceMethods<CastOpInterface>,
//      DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//      Pure,
//      SameOperandsAndResultShape
//   ]> {
//   let summary = "shape cast operation";
//   let description = [{
//     The "cast" operation converts a tensor from one type to an equivalent type
//     without changing any data elements. The source and destination types must
//     both be tensor types with the same element type. If both are ranked, then
//     shape is required to match. The operation is invalid if converting to a
//     mismatching constant dimension.
//   }];

//   let arguments = (ins F64Tensor:$input);
//   let results = (outs F64Tensor:$output);

//   let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
// }

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : Hexir_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
    The "hexir.func" operation represents a user defined function. These are
    callable SSA-region operations that contain hexir computations.

    Example:

    ```mlir
    hexir.func @main() {
      %0 = hexir.constant dense<5.500000e+00> : tensor<f64>
      %1 = hexir.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
      hexir.print %1 : tensor<2x2xf64>
      hexir.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}



//===----------------------------------------------------------------------===//
// CallTIROp
//===----------------------------------------------------------------------===//

def CallTIROp : Hexir_Op<"call_tir", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "call a hextir.prim_func, materializing its result as a tensor";
  let description = [{
    The bridge from the graph level down to the kernel level, and the only way
    to reach `hextir`.

    The two levels disagree about semantics on purpose: `hexir` is on tensors
    with value semantics, while a `hextir.prim_func` is destination-passing over
    buffers and returns nothing. `call_tir` is where that gap is crossed -- it
    names the result *type* here, and bufferization later turns that type into
    the destination buffer argument the callee actually expects.

    So the callee takes one buffer per argument, plus one for the destination:

    ```mlir
    %r = hexir.call_tir @matmul(%a, %b)
         : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>

    hextir.prim_func @matmul(%a: memref<2x2xf64>,
                             %b: memref<2x2xf64>,
                             %dst: memref<2x2xf64>) { ... }
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$args);
  let results = (outs F64Tensor:$result);

  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:` functional-type($args, $result)
  }];
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def PrintOp : Hexir_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  // We also allow a F64MemRef to enable interop during partial lowering.
  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

// def ReshapeOp : Hexir_Op<"reshape", [Pure]> {
//   let summary = "tensor reshape operation";
//   let description = [{
//     Reshape operation is transforming its input tensor into a new tensor with
//     the same number of elements but different shapes. For example:

//     ```mlir
//        %0 = hexir.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
//     ```
//   }];

//   let arguments = (ins F64Tensor:$input);

//   let assemblyFormat = [{
//     `(` $input `:` type($input) `)` attr-dict `to` type(results)
//   }];

//   // Enable registering canonicalization patterns with this operation.
//   let hasCanonicalizer = 1;

//   // We expect that the reshape operation returns a statically shaped tensor.
//   let results = (outs StaticShapeTensorOf<[F64]>);
// }


// //===----------------------------------------------------------------------===//
// // TransposeOp
// //===----------------------------------------------------------------------===//

// def TransposeOp : Hexir_Op<"transpose",
//     [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
//   let summary = "transpose operation";

//   let arguments = (ins F64Tensor:$input);
//   let results = (outs F64Tensor);

//   let assemblyFormat = [{
//     `(` $input `:` type($input) `)` attr-dict `to` type(results)
//   }];

//   // Enable registering canonicalization patterns with this operation.
//   let hasCanonicalizer = 1;

//   // Allow building a TransposeOp with from the input operand.
//   let builders = [
//     OpBuilder<(ins "Value":$input)>
//   ];

//   // Indicate that additional verification for this operation is necessary.
//   let hasVerifier = 1;
// }

#endif // HEXIR_OPS
