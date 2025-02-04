#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>
#include <stdint.h>
#include <sys/stat.h>


#ifndef INDEX_BITWIDTH
#error "INDEX_BITWIDTH is not defined!"
#endif

#if INDEX_BITWIDTH == 32
typedef uint32_t index_t;
#elif INDEX_BITWIDTH == 64
typedef uint64_t index_t;
#else
#error "Unsupported INDEX_BITWIDTH"
#endif

uint64_t get_uint64(const char *arg);
struct shm_info open_and_map_shm(const char *shm, bool is_read_only);

struct shm_info {
    void *ptr;
    struct stat sb;
};

typedef struct {
  void *allocated;
  void *aligned;
  index_t offset;
  index_t sizes;
  index_t strides;
}  memref_descriptor_t;

#define MEMREF_DESC_ARGS(desc) \
    (desc).allocated, (desc).aligned, (desc).offset, (desc).sizes, (desc).strides

#endif
