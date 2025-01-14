#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes;
  int64_t strides;
}  memref_descriptor_t;

#define MEMREF_DESC_ARGS(desc) \
    (desc).allocated, (desc).aligned, (desc).offset, (desc).sizes, (desc).strides

extern memref_descriptor_t spmv(void *, void *, int64_t, int64_t, int64_t,
                                void *, void *, int64_t, int64_t, int64_t,
                                void *, void *, int64_t, int64_t, int64_t,
                                void *, void *, int64_t, int64_t, int64_t,
                                void *, void *, int64_t, int64_t, int64_t);

struct shm_info {
    void *ptr;
    struct stat sb;
};

struct shm_info open_and_map_shm(const char *shm, bool is_read_only, bool is_hugetlbfs) {
    int shm_fd = -1;

    // Open the shared memory object
    if (is_hugetlbfs) {
       shm_fd = open(shm, O_RDWR);
       if (shm_fd == -1) {
           perror("open");
           exit(EXIT_FAILURE);
    }
    }
    else {
        shm_fd = shm_open(shm, is_read_only ? O_RDONLY : O_RDWR, 0666);
        if (shm_fd == -1) {
            perror("shm_open");
            exit(EXIT_FAILURE);
        }
    }

    // Map the shared memory object
    struct stat sb;
    if (fstat(shm_fd, &sb) == -1) {
        perror("fstat");
        close(shm_fd);
        exit(EXIT_FAILURE);
    }

    void *ptr = mmap(0, sb.st_size, is_read_only ? PROT_READ : PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        close(shm_fd);
        exit(EXIT_FAILURE);
    }

    // Close the file descriptor after mapping the memory
    close(shm_fd);

    // Create a ShmInfo struct to hold both values and return it
    struct shm_info result = {ptr, sb};
    return result;
}

uint64_t get_uint64(const char *arg) {

    // Convert the command-line argument (string) to an integer using strtol
    char *endptr;
    errno = 0; // To distinguish success/failure after call
    uint64_t num = strtoull(arg, &endptr, 10);

    // Check for conversion errors and out-of-range values
    if ( (errno == ERANGE && num == UINT64_MAX) || (errno != 0 && num == 0) ) {
        perror("strtoull");
        exit(EXIT_FAILURE);
    }

    // Check if there are any non-numeric characters in the input
    if (*endptr != '\0') {
        printf("Invalid integer: %s\n", arg);
        exit(EXIT_FAILURE);
    }

    return num;
}

int main(int argc, char **argv) {

    if (argc < 9) {
        fprintf(stderr, "Usage: %s <rows> <cols> <nnz> <vec> <mat.data> <mat.indices> <mat.indptr> <res>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    uint64_t rows = get_uint64(argv[1]);
    uint64_t cols = get_uint64(argv[2]);
    uint64_t nnz = get_uint64(argv[3]);
    struct shm_info vec = open_and_map_shm(argv[4], true, true);
    struct shm_info mat_data = open_and_map_shm(argv[5], true, false);
    struct shm_info mat_indptr = open_and_map_shm(argv[6], true, false);
    struct shm_info mat_indices = open_and_map_shm(argv[7], true, false);
    struct shm_info res = open_and_map_shm(argv[8], false, false);


    memref_descriptor_t res_d = {
        .allocated = res.ptr,
        .aligned = res.ptr,
        .offset = 0,
        .sizes = rows,
        .strides = 1
    };

    memref_descriptor_t mat_indptr_d = {
        .allocated = mat_indptr.ptr,
        .aligned = mat_indptr.ptr,
        .offset = 0,
        .sizes = rows + 1,
        .strides = 1
    };

    memref_descriptor_t mat_indices_d = {
        .allocated = mat_indices.ptr,
        .aligned = mat_indices.ptr,
        .offset = 0,
        .sizes = nnz,
        .strides = 1
    };

    memref_descriptor_t mat_data_d = {
        .allocated = mat_data.ptr,
        .aligned = mat_data.ptr,
        .offset = 0,
        .sizes = nnz,
        .strides = 1
    };

    memref_descriptor_t vec_d = {
        .allocated = vec.ptr,
        .aligned = vec.ptr,
        .offset = 0,
        .sizes = cols,
        .strides = 1
    };

    (void)spmv(MEMREF_DESC_ARGS(res_d), MEMREF_DESC_ARGS(mat_indptr_d), MEMREF_DESC_ARGS(mat_indices_d), MEMREF_DESC_ARGS(mat_data_d), MEMREF_DESC_ARGS(vec_d));

    // Clean up
    munmap(vec.ptr, vec.sb.st_size);
    munmap(mat_data.ptr, mat_data.sb.st_size);
    munmap(mat_indices.ptr, mat_indices.sb.st_size);
    munmap(mat_indptr.ptr, mat_indptr.sb.st_size);
    munmap(res.ptr, res.sb.st_size);

    return 0;
}
