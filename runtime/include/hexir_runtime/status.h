//===- status.h - Runtime status codes ---------------------------*- C -*-===//
#ifndef HEXIR_RUNTIME_STATUS_H
#define HEXIR_RUNTIME_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  HEXIR_OK = 0,
  HEXIR_ERROR_INVALID_ARGUMENT,
  HEXIR_ERROR_NOT_FOUND,
  HEXIR_ERROR_OUT_OF_MEMORY,
  HEXIR_ERROR_UNIMPLEMENTED,
  HEXIR_ERROR_INVALID_MODULE,
  HEXIR_ERROR_IO,
  HEXIR_ERROR_DEVICE,
} hexir_status_t;

const char *hexir_status_string(hexir_status_t status);

#ifdef __cplusplus
}
#endif
#endif // HEXIR_RUNTIME_STATUS_H
