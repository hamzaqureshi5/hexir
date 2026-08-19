//===- module.c - Module loading ------------------------------------------===//
//
// mmaps the file and validates the header + section table. Nothing is copied:
// section pointers point into the mapping, which is what lets a large RODATA
// section cost nothing to "load".
//
//===----------------------------------------------------------------------===//

#include "hexir_runtime/module.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct hexir_module_t {
  void *mapping; /* kept non-const purely so munmap does not need a cast */
  const uint8_t *base;
  size_t size;
  const hexir_module_header_t *header;
  const hexir_section_entry_t *sections;
};

const char *hexir_section_kind_string(uint32_t kind) {
  switch (kind) {
  case HEXIR_SECTION_SYMBOLS: return "symbols";
  case HEXIR_SECTION_PROGRAM: return "program";
  case HEXIR_SECTION_RODATA: return "rodata";
  case HEXIR_SECTION_EXECUTABLES: return "executables";
  }
  return "unknown";
}

static hexir_status_t validate(const hexir_module_t *m) {
  if (m->size < sizeof(hexir_module_header_t))
    return HEXIR_ERROR_INVALID_MODULE;
  if (memcmp(m->header->magic, HEXIR_MODULE_MAGIC, HEXIR_MODULE_MAGIC_SIZE) != 0)
    return HEXIR_ERROR_INVALID_MODULE;
  if (m->header->version != HEXIR_MODULE_VERSION)
    return HEXIR_ERROR_INVALID_MODULE;

  /* The section table must fit ... */
  size_t table_bytes =
      (size_t)m->header->section_count * sizeof(hexir_section_entry_t);
  if (sizeof(hexir_module_header_t) + table_bytes > m->size)
    return HEXIR_ERROR_INVALID_MODULE;

  /* ... and so must every payload it points at. Offsets come from a file on
     disk, so they are untrusted: check before handing out any pointer. */
  for (uint32_t i = 0; i < m->header->section_count; ++i) {
    const hexir_section_entry_t *s = &m->sections[i];
    if (s->offset > m->size || s->size > m->size - s->offset)
      return HEXIR_ERROR_INVALID_MODULE;
  }
  return HEXIR_OK;
}

hexir_status_t hexir_module_load_file(const char *path, hexir_module_t **out) {
  if (!path || !out)
    return HEXIR_ERROR_INVALID_ARGUMENT;
  *out = NULL;

  int fd = open(path, O_RDONLY);
  if (fd < 0)
    return HEXIR_ERROR_IO;

  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    close(fd);
    return HEXIR_ERROR_IO;
  }

  void *base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (base == MAP_FAILED)
    return HEXIR_ERROR_IO;

  hexir_module_t *m = calloc(1, sizeof(*m));
  if (!m) {
    munmap(base, (size_t)st.st_size);
    return HEXIR_ERROR_OUT_OF_MEMORY;
  }
  m->mapping = base;
  m->base = (const uint8_t *)base;
  m->size = (size_t)st.st_size;
  m->header = (const hexir_module_header_t *)base;
  m->sections =
      (const hexir_section_entry_t *)(m->base + sizeof(hexir_module_header_t));

  hexir_status_t status = validate(m);
  if (status != HEXIR_OK) {
    hexir_module_release(m);
    return status;
  }
  *out = m;
  return HEXIR_OK;
}

void hexir_module_release(hexir_module_t *module) {
  if (!module)
    return;
  if (module->mapping)
    munmap(module->mapping, module->size);
  free(module);
}

uint32_t hexir_module_version(const hexir_module_t *m) {
  return m ? m->header->version : 0u;
}

uint32_t hexir_module_section_count(const hexir_module_t *m) {
  return m ? m->header->section_count : 0u;
}

const hexir_section_entry_t *hexir_module_section_at(const hexir_module_t *m,
                                                     uint32_t index) {
  if (!m || index >= m->header->section_count)
    return NULL;
  return &m->sections[index];
}

hexir_status_t hexir_module_section(const hexir_module_t *m,
                                    hexir_section_kind_t kind,
                                    const void **data, uint64_t *size) {
  if (!m || !data || !size)
    return HEXIR_ERROR_INVALID_ARGUMENT;
  for (uint32_t i = 0; i < m->header->section_count; ++i) {
    if (m->sections[i].kind == (uint32_t)kind) {
      *data = m->base + m->sections[i].offset;
      *size = m->sections[i].size;
      return HEXIR_OK;
    }
  }
  *data = NULL;
  *size = 0;
  return HEXIR_ERROR_NOT_FOUND;
}
