#include "hexir_runtime/status.h"

const char *hexir_status_string(hexir_status_t status) {
  switch (status) {
  case HEXIR_OK: return "ok";
  case HEXIR_ERROR_INVALID_ARGUMENT: return "invalid argument";
  case HEXIR_ERROR_NOT_FOUND: return "not found";
  case HEXIR_ERROR_OUT_OF_MEMORY: return "out of memory";
  case HEXIR_ERROR_UNIMPLEMENTED: return "unimplemented";
  case HEXIR_ERROR_INVALID_MODULE: return "invalid module";
  case HEXIR_ERROR_IO: return "io error";
  case HEXIR_ERROR_DEVICE: return "device error";
  }
  return "unknown";
}
