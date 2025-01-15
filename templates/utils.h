#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>
#include <stdint.h>
#include <sys/stat.h>

uint64_t get_uint64(const char *arg);
struct shm_info open_and_map_shm(const char *shm, bool is_read_only);

struct shm_info {
    void *ptr;
    struct stat sb;
};

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes;
  int64_t strides;
}  memref_descriptor_t;

#define MEMREF_DESC_ARGS(desc) \
    (desc).allocated, (desc).aligned, (desc).offset, (desc).sizes, (desc).strides

#endif
