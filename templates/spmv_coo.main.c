#include "utils.h"

#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>


extern memref_descriptor_t spmv(void *, void *, int64_t, int64_t, int64_t,  // vec
                                void *, void *, int64_t, int64_t, int64_t,  // mat.pos
                                void *, void *, int64_t, int64_t, int64_t,  // mat.row
                                void *, void *, int64_t, int64_t, int64_t,  // mat.col
                                void *, void *, int64_t, int64_t, int64_t,  // mat.data
                                void *, void *, int64_t, int64_t, int64_t); // res

int main(int argc, char **argv) {

    if (argc < 10) {
        fprintf(stderr, "Usage: %s <rows> <cols> <nnz> <vec> <mat.pos> <mat.row> <mat.col> <mat.data> <res>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    uint64_t rows = get_uint64(argv[1]);
    uint64_t cols = get_uint64(argv[2]);
    uint64_t nnz = get_uint64(argv[3]);
    struct shm_info vec = open_and_map_shm(argv[4], true);
    struct shm_info mat_pos = open_and_map_shm(argv[5], true);
    struct shm_info mat_row = open_and_map_shm(argv[6], true);
    struct shm_info mat_col = open_and_map_shm(argv[7], true);
    struct shm_info mat_data = open_and_map_shm(argv[8], true);
    struct shm_info res = open_and_map_shm(argv[9], false);

    memref_descriptor_t res_d = {
        .allocated = res.ptr,
        .aligned = res.ptr,
        .offset = 0,
        .sizes = rows,
        .strides = 1
    };

    memref_descriptor_t mat_pos_d = {
        .allocated = mat_pos.ptr,
        .aligned = mat_pos.ptr,
        .offset = 0,
        .sizes = 2,
        .strides = 1
    };

    memref_descriptor_t mat_row_d = {
        .allocated = mat_row.ptr,
        .aligned = mat_row.ptr,
        .offset = 0,
        .sizes = nnz,
        .strides = 1
    };

    memref_descriptor_t mat_col_d = {
        .allocated = mat_col.ptr,
        .aligned = mat_col.ptr,
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

    struct timespec start_ts;
    clock_gettime(CLOCK_MONOTONIC, &start_ts);

    (void)spmv(MEMREF_DESC_ARGS(res_d), MEMREF_DESC_ARGS(mat_pos_d), MEMREF_DESC_ARGS(mat_row_d), MEMREF_DESC_ARGS(mat_col_d), MEMREF_DESC_ARGS(mat_data_d), MEMREF_DESC_ARGS(vec_d));

    struct timespec end_ts;
    clock_gettime(CLOCK_MONOTONIC, &end_ts);

    long seconds = end_ts.tv_sec - start_ts.tv_sec;
    long nanoseconds = end_ts.tv_nsec - start_ts.tv_nsec;
    printf("Exec time: %fs\n", seconds + nanoseconds*1e-9);

    // Clean up
    munmap(vec.ptr, vec.sb.st_size);
    munmap(mat_data.ptr, mat_data.sb.st_size);
    munmap(mat_pos.ptr, mat_pos.sb.st_size);
    munmap(mat_row.ptr, mat_row.sb.st_size);
    munmap(mat_col.ptr, mat_col.sb.st_size);
    munmap(res.ptr, res.sb.st_size);

    return 0;
}
