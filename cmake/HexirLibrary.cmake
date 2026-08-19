# hexir_library(<name> <sources...> [DEPENDS <tablegen targets...>])
#
# One static library per module. Every hexir library gets the same MLIR link
# set (HEXIR_MLIR_LIBS, defined in the root CMakeLists) -- the layering that
# matters here is between hexir's own modules, not against MLIR, and pinning
# exact MLIR deps per library is churn with no benefit at this size.
function(hexir_library name)
  cmake_parse_arguments(ARG "" "" "DEPENDS" ${ARGN})
  add_library(${name} STATIC ${ARG_UNPARSED_ARGUMENTS})
  llvm_update_compile_flags(${name})
  target_include_directories(${name} PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)
  target_link_libraries(${name} PUBLIC ${HEXIR_MLIR_LIBS})
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  set_property(GLOBAL APPEND PROPERTY HEXIR_LIBS ${name})
endfunction()
