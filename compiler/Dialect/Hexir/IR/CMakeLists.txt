hexir_library(HexirDialectIR
  HexirDialect.cpp
  HexirCombine.cpp
  DEPENDS HexirOpsIncGen HexirShapeInferenceIncGen HexirCombineIncGen HexTIROpsIncGen)
