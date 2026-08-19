#ifndef BUILDER_H
#define BUILDER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;

namespace builder {

// Basic examples
func::FuncOp createMainFunction(MLIRContext &ctx, ModuleOp module);

func::FuncOp createMainFunction(MLIRContext &ctx, ModuleOp module);

func::FuncOp createAddFunction(MLIRContext &ctx, ModuleOp module);

func::FuncOp createMulFunction(MLIRContext &ctx, ModuleOp module);

// MLP examples
func::FuncOp createMLPAddFunction(MLIRContext &ctx, ModuleOp module);

func::FuncOp createMLPReluFunction(MLIRContext &ctx, ModuleOp module);

func::FuncOp createMLPLinearFunction(MLIRContext &ctx, ModuleOp module);

} // namespace builder

#endif // TOY_BUILDER_H
