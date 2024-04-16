#define _GNU_SOURCE

#include "spmv.h"

#include <time.h>

#include "hw_pref_control.h"

#define CL_SIZE_IN_COL_INDICES 8

#ifndef L1_MSHRS
#warning "L1_MSHRS not defined, using 10"
#define L1_MSHRS 10
#endif

#ifndef L2_MSHRS
#warning "L2_MSHRS not defined, using 40"
#define L2_MSHRS 40
#endif

#define PREFETCHT2  2
#define PREFETCHT0  3


static void spmv(uint64_t num_of_rows, const double *vec, const double *mat_vals, const int64_t *pos, const int64_t *crd, double *res)
{
    for (uint64_t i = 0; i < num_of_rows; i++) {

        res[i] = 0.0;
        const int64_t j_start = pos[i];
        const int64_t j_end = pos[i + 1];

        // Fill l1 mshrs by fetching l1_mshrs cache lines
        int64_t k;
#pragma clang loop unroll_count(CL_SIZE_IN_COL_INDICES)
        for (k = j_start; k < j_start + L1_MSHRS * CL_SIZE_IN_COL_INDICES; k+=CL_SIZE_IN_COL_INDICES) {
            __builtin_prefetch(&crd[k], 0, PREFETCHT2);
        }

        // Fill l2 mshrs by fetching l2_mshr elements
        for (int64_t l = j_start; l < j_start + L2_MSHRS; l++) {
            __builtin_prefetch(&vec[crd[l]], 0, PREFETCHT2);
        }

        // Assume that all capacity for MLP is now exhausted:
        // Steady state - start computation

        int64_t num_of_non_zeros_in_row = j_end - j_start;
        int64_t j_end_closest_mult_of_cl_size = j_start + (num_of_non_zeros_in_row / CL_SIZE_IN_COL_INDICES) * CL_SIZE_IN_COL_INDICES;

        for (int64_t j = j_start; j < j_end_closest_mult_of_cl_size; j+=CL_SIZE_IN_COL_INDICES, k+=CL_SIZE_IN_COL_INDICES) {

#pragma clang loop unroll_count(CL_SIZE_IN_COL_INDICES)
            for (int64_t l = 0; l < CL_SIZE_IN_COL_INDICES; l++) {
                res[i] += mat_vals[j + l] * vec[crd[j + l]];
                __builtin_prefetch(&vec[crd[j + l + L2_MSHRS]], 0, PREFETCHT2);
            }

            __builtin_prefetch(&crd[k], 0, PREFETCHT2);
        }

        for (int64_t j = j_end_closest_mult_of_cl_size; j < j_end; j++) {
            res[i] += mat_vals[j] * vec[crd[j]];
        }
    }
}

double compute(uint64_t num_of_rows, const double *vec, const double *mat_vals, const int64_t *pos, const int64_t *crd, double *res) {

    init_hw_pref_control();

    struct timespec start_ts;
    clock_gettime(CLOCK_MONOTONIC, &start_ts);

    spmv(num_of_rows, vec, mat_vals, pos, crd, res);

    struct timespec end_ts;
    clock_gettime(CLOCK_MONOTONIC, &end_ts);

    deinit_hw_pref_control();

    long seconds = end_ts.tv_sec - start_ts.tv_sec;
    long nanoseconds = end_ts.tv_nsec - start_ts.tv_nsec;
    return seconds + nanoseconds*1e-9;
}
