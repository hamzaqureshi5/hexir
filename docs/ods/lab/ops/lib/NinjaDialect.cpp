#include "ninja/NinjaDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ninja;

#include "ninja/NinjaDialect.cpp.inc"
#include "ninja/NinjaEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ninja/NinjaTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ninja/NinjaAttrs.cpp.inc"

// custom<Stars>($count): written as one `*` per unit, e.g. `***` for 3.
static ParseResult parseStars(OpAsmParser &parser, IntegerAttr &count) {
  int32_t n = 0;
  while (succeeded(parser.parseOptionalStar()))
    ++n;
  if (n == 0)
    return parser.emitError(parser.getCurrentLocation(), "expected at least one '*'");
  count = parser.getBuilder().getI32IntegerAttr(n);
  return success();
}

static void printStars(OpAsmPrinter &p, Operation *, IntegerAttr count) {
  for (int64_t i = 0; i < count.getInt(); ++i)
    p << '*';
}

#define GET_OP_CLASSES
#include "ninja/NinjaOps.cpp.inc"

void NinjaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ninja/NinjaOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "ninja/NinjaTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ninja/NinjaAttrs.cpp.inc"
      >();
}

Operation *NinjaDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  auto f = dyn_cast<FloatAttr>(value);
  if (!f || !type.isF64())
    return nullptr;
  return ConstantOp::create(builder, loc, f.getValueAsDouble());
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValueAttr(); }

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<FloatAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<FloatAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
    return {};
  return FloatAttr::get(getType(),
                        lhs.getValueAsDouble() + rhs.getValueAsDouble());
}

LogicalResult StrikeOp::verify() {
  if (getTimes() <= 0)
    return emitOpError("times must be positive, got ") << getTimes();
  return success();
}

LogicalResult StrikeOp::canonicalize(StrikeOp op, PatternRewriter &rewriter) {
  if (!op.isSingleStrike())
    return failure();
  rewriter.replaceOp(op, op.getTarget());
  return success();
}

namespace {
#include "ninja/NinjaPatterns.inc"
} // namespace

namespace {
/// x + 0.0  ==>  x
struct AddZero : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override {
    FloatAttr rhs;
    if (!matchPattern(op.getRhs(), m_Constant(&rhs)) || !rhs.getValue().isZero())
      return failure();                       // rule does not apply
    rewriter.replaceOp(op, op.getLhs());      // use x instead
    return success();
  }
};
} // namespace

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<AddSelfToStrike, AddZero>(context);
}

// ShurikenType: a shuriken needs at least one point.
LogicalResult ShurikenType::verify(function_ref<InFlightDiagnostic()> emitError,
                                   unsigned points) {
  if (points == 0)
    return emitError() << "a shuriken needs at least one point";
  return success();
}
