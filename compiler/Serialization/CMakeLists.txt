hexir_library(HexirSerialization ModuleSerializer.cpp DEPENDS HexirOpsIncGen)
# The format is defined by the runtime; the compiler emits it. Headers only --
# nothing links back, and the dependency must never point the other way.
target_include_directories(HexirSerialization PUBLIC
  ${CMAKE_SOURCE_DIR}/runtime/include)
