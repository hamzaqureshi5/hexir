//===- module.h - Deployable module container --------------------*- C -*-===//
//
// A compiled hexir program: the host program plus every device binary it
// dispatches, in one file the runtime can load without the compiler.
//
// Layout is a header, a section table, then section payloads:
//
//   [header][section entry 0..n-1][ ...payloads... ]
//
// Sections are located by absolute file offset so the whole file can be
// mmapped and RODATA used in place -- loading a large model must not mean
// reading it. That constraint is why offsets are absolute and payloads are
// 8-byte aligned.
//
// The layout is intentionally simple and versioned; it is expected to change.
//
//===----------------------------------------------------------------------===//
#ifndef HEXIR_RUNTIME_MODULE_H
#define HEXIR_RUNTIME_MODULE_H

#include <stdint.h>
#include "hexir_runtime/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HEXIR_MODULE_MAGIC "HEXIRMOD"
#define HEXIR_MODULE_MAGIC_SIZE 8
#define HEXIR_MODULE_VERSION 1u

typedef enum {
  /// Exported entry points: name, signature, shapes.
  HEXIR_SECTION_SYMBOLS = 1,
  /// The host program. Today a flat command list (alloc / copy / dispatch);
  /// a bytecode VM only becomes necessary once there is control flow.
  HEXIR_SECTION_PROGRAM = 2,
  /// Weights and constants. Used in place from the mapping, never copied.
  HEXIR_SECTION_RODATA = 3,
  /// Device binaries: CUBIN, PTX or host object code, one per kernel/target.
  HEXIR_SECTION_EXECUTABLES = 4,
} hexir_section_kind_t;

typedef struct {
  char magic[HEXIR_MODULE_MAGIC_SIZE];
  uint32_t version;
  uint32_t flags;
  uint32_t section_count;
  uint32_t reserved;
} hexir_module_header_t;

typedef struct {
  uint32_t kind;
  uint32_t flags;
  uint64_t offset; /* absolute, from the start of the file */
  uint64_t size;
} hexir_section_entry_t;

typedef struct hexir_module_t hexir_module_t;

/// Map a module file. The mapping stays alive until release, so pointers
/// handed out by hexir_module_section remain valid.
hexir_status_t hexir_module_load_file(const char *path, hexir_module_t **out);
void hexir_module_release(hexir_module_t *module);

uint32_t hexir_module_version(const hexir_module_t *module);
uint32_t hexir_module_section_count(const hexir_module_t *module);
const hexir_section_entry_t *hexir_module_section_at(const hexir_module_t *module,
                                                     uint32_t index);

/// Zero-copy: `*data` points into the mapping.
hexir_status_t hexir_module_section(const hexir_module_t *module,
                                    hexir_section_kind_t kind,
                                    const void **data, uint64_t *size);

const char *hexir_section_kind_string(uint32_t kind);

#ifdef __cplusplus
}
#endif
#endif // HEXIR_RUNTIME_MODULE_H
