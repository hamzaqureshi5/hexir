//===- BufferizableOpInterfaceImpl.h - Hexir bufferization ------*- C++ -*-===//
//
// Registers BufferizableOpInterface external models for hexir ops, so that
// One-Shot Bufferize can convert them from tensors to memrefs instead of
// bailing out with "op was not bufferized".
//
//===----------------------------------------------------------------------===//

#ifndef HEXIR_BUFFERIZABLEOPINTERFACEIMPL_H
#define HEXIR_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace hexir {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace hexir
} // namespace mlir

#endif // HEXIR_BUFFERIZABLEOPINTERFACEIMPL_H
