#include "hexir/Support/Builder.h"
#include "hexir/Dialect/Hexir/IR/HexirDialect.h"
#include "hexir/Conversion/Passes.h"
#include "hexir/Dialect/Hexir/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cassert>

#ifdef PRINT
#define PRINT
#endif
// #define PRINT

using namespace mlir;
// using namespace dbs;
using namespace mlir::hexir;

namespace builder {
/**
 * createMainFunction
 *
 * Construct a top-level "main" function inside the provided MLIR module.
 *
 * Behavior:
 * - Creates a func::FuncOp named "main" with an empty function type (no
 *   arguments, no results).
 * - Adds the function to the supplied ModuleOp and inserts a new entry block.
 * - Establishes an OpBuilder insertion point at the start of the entry block.
 * - Emits two f32 constants (1.0 and 2.0) into the function body.
 * - Emits a func::ReturnOp to terminate the function.
 *
 * Conditional behavior (enabled when PRINT is defined at compile time):
 * - Declares a helper function "print_f32" (f32 -> void) at module scope and
 *   marks it private.
 * - Attempts to emit calls to "print_f32" with the results of intermediate
 *   operations (intended to print computed f32 values). Note: those
 * intermediate values (e.g. add, mul) must exist in the function body for the
 * print calls to be valid.
 *
 * Parameters:
 * - ctx:   The MLIRContext used to construct operations and types.
 * - module: The ModuleOp into which the "main" function will be inserted.
 *
 * Side effects:
 * - Mutates the provided ModuleOp by appending the newly created FuncOp and,
 *   when PRINT is defined, by inserting a private "print_f32" declaration.
 * - Creates operations and IR in the provided MLIRContext.
 *
 * Returns:
 * - The created func::FuncOp corresponding to "main".
 *
 * Notes:
 * - All created operations use builder.getUnknownLoc() for locations.
 * - The function currently has no arguments and returns nothing; callers that
 *   expect different signatures should modify the function type accordingly.
 */
func::FuncOp createMainFunction(MLIRContext &ctx, ModuleOp module) {

  mlir::OpBuilder builder(&ctx);

  [[maybe_unused]] auto f32 = builder.getF32Type();
  auto funcType = builder.getFunctionType({}, {});

  // auto func =
  //     func::FuncOp::create(builder,builder.getUnknownLoc(), "main",
  //     funcType);
  auto func =
      mlir::func::FuncOp::create(builder.getUnknownLoc(), "main", funcType);

  module.push_back(func);

  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

#ifdef PRINT
  // ---- Declare print function ----
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());

    auto printType = builder.getFunctionType({f32}, {});
    builder
        .create<func::FuncOp>(builder.getUnknownLoc(), "print_f32", printType)
        .setPrivate();
  }
#endif

  [[maybe_unused]] Value c1 = arith::ConstantOp::create(
      builder, builder.getUnknownLoc(), builder.getF32FloatAttr(1.0));
  [[maybe_unused]] Value c2 = arith::ConstantOp::create(
      builder, builder.getUnknownLoc(), builder.getF32FloatAttr(2.0));

  //  [[maybe_unused]] Value add = builder
  //  .create<func::CallOp>(builder.getUnknownLoc(), "my_mul",
  //  builder.getF32Type(), ValueRange{c1, c2}) ->getResult(0);
  //   Value sum =
  //   builder.create<arith::AddFOp::create(builder,builder.getUnknownLoc(), c1,
  //   c2);

#ifdef PRINT
  // Print mul result
  builder.create<func::CallOp>(builder.getUnknownLoc(), "print_f32",
                               TypeRange{}, ValueRange{add});
#endif

  // //  --- Call my_mul ---
  // Value c3 = arith::ConstantOp::create(builder,builder.getUnknownLoc(),
  //                                              builder.getF32FloatAttr(3.0));
  // Value c4 = arith::ConstantOp::create(builder,builder.getUnknownLoc(),
  //                                              builder.getF32FloatAttr(4.0));

  // [[maybe_unused]] Value mul =
  //     builder
  //         .create<func::CallOp>(builder.getUnknownLoc(), "my_mul",
  //                               builder.getF32Type(), ValueRange{c3, c4})
  //         ->getResult(0);
#ifdef PRINT
  // Print mul result
  builder.create<func::CallOp>(builder.getUnknownLoc(), "print_f32",
                               TypeRange{}, ValueRange{mul});
#endif

  // auto c5 = arith::ConstantOp::create(builder,builder.getUnknownLoc(),
  //                                             builder.getF32FloatAttr(1.0));
  // auto c6 = arith::ConstantOp::create(builder,builder.getUnknownLoc(),
  //                                             builder.getF32FloatAttr(2.0));

  // // Use your custom dialect op
  // auto myAdd =
  // builder.create<mlir::dbs::Add::create(builder,builder.getUnknownLoc(), c1,
  // c2);

  func::ReturnOp::create(builder, builder.getUnknownLoc());
  // mlir::func::ReturnOp::create(builder.getUnknownLoc());
  return func;
}

// ---------------------------------------------
// Create MLIR function: mul(a : f32, b : f32) -> f32
// ---------------------------------------------
func::FuncOp createMulFunction(MLIRContext &ctx, ModuleOp module) {
  OpBuilder builder(&ctx);

  auto f32 = builder.getF32Type();
  auto funcType = builder.getFunctionType({f32, f32}, {f32});

  auto func = func::FuncOp::create(builder, builder.getUnknownLoc(), "my_mul",
                                   funcType);

  func.setVisibility(mlir::SymbolTable::Visibility::Public);

  module.push_back(func);

  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  Value a = entry->getArgument(0);
  Value b = entry->getArgument(1);

  Value prod = arith::MulFOp::create(builder, builder.getUnknownLoc(), a, b);

  func::ReturnOp::create(builder, builder.getUnknownLoc(), prod);
  return func;
}

// ---------------------------------------------
// Create MLIR function: add(a : f32, b : f32) -> f32
// ---------------------------------------------

func::FuncOp createAddFunction(MLIRContext &ctx, ModuleOp module) {
  OpBuilder builder(&ctx);

  auto f32 = builder.getF32Type();
  auto funcType = builder.getFunctionType({f32, f32}, {f32});

  auto func = func::FuncOp::create(builder, builder.getUnknownLoc(), "my_add",
                                   funcType);

  func.setVisibility(mlir::SymbolTable::Visibility::Public);

  module.push_back(func);

  // Create entry block
  mlir::Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value a = entry->getArgument(0);
  Value b = entry->getArgument(1);

  Value sum = arith::AddFOp::create(builder, builder.getUnknownLoc(), a, b);

  func::ReturnOp::create(builder, builder.getUnknownLoc(), sum);
  return func;
}

func::FuncOp createMLPAddFunction(MLIRContext &ctx,
                                  ModuleOp module) { // WORKING ON THIS
  OpBuilder builder(&ctx);
  Location loc = builder.getUnknownLoc();

  // auto f32 = builder.getF32Type();
  auto f64 = builder.getF64Type();

  auto rankedtensorf64Ty = RankedTensorType::get({2}, f64);
  // auto rankedtensorf32Ty = RankedTensorType::get({2}, f32);

  auto funcType = builder.getFunctionType({}, {});

  auto func =
      func::FuncOp::create(builder, builder.getUnknownLoc(), "main", funcType);

  func.setVisibility(mlir::SymbolTable::Visibility::Public);

  module.push_back(func);
  // Create entry block
  mlir::Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  llvm::SmallVector<llvm::APFloat, 2> vals1 = {llvm::APFloat(5.0),
                                               llvm::APFloat(7.0)};
  llvm::SmallVector<llvm::APFloat, 2> vals2 = {llvm::APFloat(10.0),
                                               llvm::APFloat(5.0)};

  auto denseAttr1 = mlir::DenseElementsAttr::get(rankedtensorf64Ty, vals1);
  auto denseAttr2 = mlir::DenseElementsAttr::get(rankedtensorf64Ty, vals2);

  Value c1 =
      hexir::ConstantOp::create(builder, loc, rankedtensorf64Ty, denseAttr1);

  Value c2 =
      hexir::ConstantOp::create(builder, loc, rankedtensorf64Ty, denseAttr2);

  Value add = hexir::AddOp::create(builder, loc, rankedtensorf64Ty, c1, c2);
  // Value add =
  //     builder.create < hexir::ReluOp::create(builder, loc, rankedtensorf64Ty,
  //     c1);

  hexir::PrintOp::create(builder, builder.getUnknownLoc(), c1);
  hexir::PrintOp::create(builder, builder.getUnknownLoc(), c2);
  hexir::PrintOp::create(builder, builder.getUnknownLoc(), add);

  func::ReturnOp::create(builder, loc);

  return func;
}

func::FuncOp createMLPLinearFunction(MLIRContext &ctx,
                                     ModuleOp module) { // WORKING ON THIS

  OpBuilder builder(&ctx);
  Location loc = builder.getUnknownLoc();

  auto f64 = builder.getF64Type();

  auto rankedtensorfTy = RankedTensorType::get({2, 2}, f64);

  auto funcType = builder.getFunctionType({}, {});

  auto func =
      func::FuncOp::create(builder, builder.getUnknownLoc(), "main", funcType);

  func.setVisibility(mlir::SymbolTable::Visibility::Public);

  module.push_back(func);
  // Create entry block
  mlir::Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  llvm::SmallVector<llvm::APFloat, 2> vals1D = {llvm::APFloat(3.0),
                                                llvm::APFloat(1.0)};
  llvm::SmallVector<llvm::APFloat, 2> vals1 = {
      llvm::APFloat(3.0), llvm::APFloat(1.0), llvm::APFloat(2.0),
      llvm::APFloat(2.0)};
  llvm::SmallVector<llvm::APFloat, 2> vals2 = {
      llvm::APFloat(1.0), llvm::APFloat(5.0), llvm::APFloat(5.0),
      llvm::APFloat(2.0)};

  int64_t N = vals1D.size();
  auto tensor1DTy = RankedTensorType::get({1, N}, f64); // Batch is 1
  // auto tensor1DTy = RankedTensorType::get({1,N}, f64); // Batch is 1
  auto denseAttr1D = mlir::DenseElementsAttr::get(tensor1DTy, vals1D);
  [[maybe_unused]] Value c1D =
      hexir::ConstantOp::create(builder, loc, tensor1DTy, denseAttr1D);

  auto denseAttr1 = mlir::DenseElementsAttr::get(rankedtensorfTy, vals1);
  auto denseAttr2 = mlir::DenseElementsAttr::get(rankedtensorfTy, vals2);

  Value c1 =
      hexir::ConstantOp::create(builder, loc, rankedtensorfTy, denseAttr1);
  Value c2 =
      hexir::ConstantOp::create(builder, loc, rankedtensorfTy, denseAttr2);
  // hexir::PrintOp::create(builder, loc, c1);
  // hexir::PrintOp::create(builder, loc, c2);

  Value lin = hexir::LinearOp::create(builder, loc, rankedtensorfTy, c1, c2);

  hexir::PrintOp::create(builder, loc, lin);

  //Value relu = hexir::ReluOp::create(builder, loc, rankedtensorfTy, lin);

  // Value softmax = builder.create<hexir::SoftmaxOp::create(builder,loc,
  // tensor1DTy, c1D);

  //hexir::PrintOp::create(builder, loc, relu);

  mlir::func::ReturnOp::create(builder, loc);

  return func;
}

// func::FuncOp createMLPAddTOSAFunction(MLIRContext &ctx,
//                                       ModuleOp module) { // WORKING ON THIS
//   OpBuilder builder(&ctx);
//   Location loc = builder.getUnknownLoc();

//   auto f32 = builder.getF32Type();
//   auto f64 = builder.getF64Type();

//   // auto rankedtensor32Ty = RankedTensorType::get({2}, f32);
//   auto rankedtensor64Ty = RankedTensorType::get({2}, f64);

//   auto funcType = builder.getFunctionType({}, {});

//   auto func =
//       func::FuncOp::create(builder,builder.getUnknownLoc(), "main",
//       funcType);

//   func.setVisibility(mlir::SymbolTable::Visibility::Public);

//   module.push_back(func);
//   // Create entry block
//   mlir::Block *entry = func.addEntryBlock();
//   builder.setInsertionPointToStart(entry);

//   // llvm::SmallVector<llvm::APFloat, 2> vals1 = {llvm::APFloat(5.0),
//   //                                              llvm::APFloat(7.0)};
//   // llvm::SmallVector<llvm::APFloat, 2> vals2 = {llvm::APFloat(10.0),
//   //                                              llvm::APFloat(5.0)};

//   // auto denseAttr1 = mlir::DenseElementsAttr::get(rankedtensorTy, vals1);
//   // auto denseAttr2 = mlir::DenseElementsAttr::get(rankedtensorTy, vals2);
//   std::vector<float> vals1 = {5, 7};
//   std::vector<float> vals2 = {10, 5};

//   // Convert to FloatAttr
//   llvm::SmallVector<mlir::Attribute, 2> attrs1;
//   for (auto v : vals1)
//     attrs1.push_back(builder.getFloatAttr(f64, v));

//   // llvm::SmallVector<mlir::Attribute, 2> attrs2;
//   // for (auto v : vals2)
//   //   attrs2.push_back(builder.getFloatAttr(i32, v));

//   auto denseAttr1 = mlir::DenseElementsAttr::get(rankedtensor64Ty, attrs1);
//   //  auto denseAttr2 = mlir::DenseElementsAttr::get(rankedtensorTy, attrs2);

//   /////////////////////////////
//   // 1) f64 tensor constant
//   auto denseAttr5 = DenseElementsAttr::get(rankedtensor64Ty, attrs1);

//   // Value c64 =
//   //     hexir::ConstantOp::create(builder,loc, rankedtensor64Ty,
//   denseAttr5);

//   // // 2) cast f64 -> f32 (tensor-level)
//   // Value c32 =
//   //     builder.create<arith::TruncFOp>(loc, rankedtensor32Ty, c64);

//   // // 3) use f32 result
//   // hexir::PrintOp::create(builder,loc, c32);

//   /////////////////////////////

//   // Value c1 = hexir::ConstantOp::create(builder,loc, rankedtensor32Ty,
//   // denseAttr1);

//   // Value c2 = builder.create<tosa::ConstOp::create(builder,loc,
//   rankedtensorTy, denseAttr2);

//   // Value add = builder.create<tosa::AddOp::create(builder,loc,
//   rankedtensorTy, c1, c2);

//   // hexir::PrintOp::create(builder,builder.getUnknownLoc(), c32);
//   //   hexir::PrintOp::create(builder,builder.getUnknownLoc(), c22);
//   //   hexir::PrintOp::create(builder,builder.getUnknownLoc(), add);

//   auto t32 = RankedTensorType::get({2}, f32);
//   auto t64 = RankedTensorType::get({2}, f64);

//   // Build f32 attrs
//   llvm::SmallVector<Attribute, 2> attrs64;
//   for (float v : vals1)
//     attrs64.push_back(builder.getFloatAttr(f64, v));

//   auto dense32 = DenseElementsAttr::get(t64, attrs64);
//   Value c32 = hexir::ConstantOp::create(builder,loc, t32, dense32);
//   hexir::PrintOp::create(builder,builder.getUnknownLoc(), c32);
//   builder.create<func::ReturnOp::create(builder,loc);

//   return func;
// }

func::FuncOp createMLPReluFunction(MLIRContext &ctx,
                                   ModuleOp module) { // WORKING ON THIS
  OpBuilder builder(&ctx);

  auto f64 = builder.getF64Type();

  auto rankedtensorTy = RankedTensorType::get({4}, f64);
  [[maybe_unused]] auto unrankedtensorTy = UnrankedTensorType::get(f64);

  auto funcType = builder.getFunctionType({}, {});

  auto func =
      func::FuncOp::create(builder, builder.getUnknownLoc(), "main", funcType);

  func.setVisibility(mlir::SymbolTable::Visibility::Public);

  module.push_back(func);
  // Create entry block
  mlir::Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  llvm::SmallVector<llvm::APFloat, 4> vals1 = {
      llvm::APFloat(5.0), llvm::APFloat(-7.0), llvm::APFloat(7.0),
      llvm::APFloat(10.0)};

  auto denseAttr1 = mlir::DenseElementsAttr::get(rankedtensorTy, vals1);

  Value c11 = hexir::ConstantOp::create(builder, builder.getUnknownLoc(),
                                        rankedtensorTy, denseAttr1);

  Value relu = hexir::ReluOp::create(builder, builder.getUnknownLoc(),
                                     rankedtensorTy, c11);

  hexir::PrintOp::create(builder, builder.getUnknownLoc(), c11);

  hexir::PrintOp::create(builder, builder.getUnknownLoc(), relu);

  func::ReturnOp::create(builder, builder.getUnknownLoc());
  return func;
}

} // namespace builder
