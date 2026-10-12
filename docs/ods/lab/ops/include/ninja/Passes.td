include "mlir/Pass/PassBase.td"

def NinjaCountStrikes : Pass<"ninja-count-strikes", "func::FuncOp"> {
  let summary = "count the ninja.strike ops in each function";
  let options = [
    Option<"verbose", "verbose", "bool", /*default=*/"false",
           "also print the function name">
  ];
}
