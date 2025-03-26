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

extern memref_descriptor_t spmm(void *, void *, index_t, index_t, index_t, index_t, index_t,
                                void *, void *, index_t, index_t, index_t,
                                void *, void *, index_t, index_t, index_t,
                                void *, void *, index_t, index_t, index_t,
                                void *, void *, index_t, index_t, index_t, index_t, index_t);

int main(int argc, char **argv) {

    if (argc < 10) {
        fprintf(stderr, "Usage: %s <i> <j> <k> <nnz> <vec> <sp_mat.indices> <sp_mat.indptr> <sp_mat.data> <res>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    uint64_t i = get_uint64(argv[1]);
    uint64_t j = get_uint64(argv[2]);
    uint64_t k = get_uint64(argv[3]);
    uint64_t nnz = get_uint64(argv[4]);
    struct shm_info dense_mat = open_and_map_shm(argv[5], true);
    struct shm_info sp_mat_indptr = open_and_map_shm(argv[6], true);
    struct shm_info sp_mat_indices = open_and_map_shm(argv[7], true);
    struct shm_info sp_mat_data = open_and_map_shm(argv[8], true);
    struct shm_info res = open_and_map_shm(argv[9], false);

    mat_memref_descriptor_t res_d = {
        .allocated = res.ptr,
        .aligned = res.ptr,
        .offset = 0,
        .rows = i,
        .cols = k,
        .stride_rows = k,
        .stride_cols = 1
    };

    memref_descriptor_t sp_mat_indptr_d = {
        .allocated = sp_mat_indptr.ptr,
        .aligned = sp_mat_indptr.ptr,
        .offset = 0,
        .sizes = i + 1,
        .strides = 1
    };

    memref_descriptor_t sp_mat_indices_d = {
        .allocated = sp_mat_indices.ptr,
        .aligned = sp_mat_indices.ptr,
        .offset = 0,
        .sizes = nnz,
        .strides = 1
    };

    memref_descriptor_t sp_mat_data_d = {
        .allocated = sp_mat_data.ptr,
        .aligned = sp_mat_data.ptr,
        .offset = 0,
        .sizes = nnz,
        .strides = 1
    };

    mat_memref_descriptor_t dense_mat_d = {
        .allocated = dense_mat.ptr,
        .aligned = dense_mat.ptr,
        .offset = 0,
        .rows = j,
        .cols = k,
        .stride_rows = k,
        .stride_cols = 1
    };

    struct timespec start_ts;
    clock_gettime(CLOCK_MONOTONIC, &start_ts);

    (void)spmm(MAT_MEMREF_DESC_ARGS(res_d), MEMREF_DESC_ARGS(sp_mat_indptr_d), MEMREF_DESC_ARGS(sp_mat_indices_d), MEMREF_DESC_ARGS(sp_mat_data_d), MAT_MEMREF_DESC_ARGS(dense_mat_d));

    struct timespec end_ts;
    clock_gettime(CLOCK_MONOTONIC, &end_ts);

    long seconds = end_ts.tv_sec - start_ts.tv_sec;
    long nanoseconds = end_ts.tv_nsec - start_ts.tv_nsec;
    printf("Exec time: %fs\n", seconds + nanoseconds*1e-9);

    // Clean up
    munmap(dense_mat.ptr, dense_mat.sb.st_size);
    munmap(sp_mat_data.ptr, sp_mat_data.sb.st_size);
    munmap(sp_mat_indices.ptr, sp_mat_indices.sb.st_size);
    munmap(sp_mat_indptr.ptr, sp_mat_indptr.sb.st_size);
    munmap(res.ptr, res.sb.st_size);

    return 0;
}
