#define _GNU_SOURCE

#include "spmv.h"

#include <time.h>


#ifndef L2_MSHRS
#warning "L2_MSHRS not defined, using 40"
#define L2_MSHRS 40
#endif

#define CL_SIZE_IN_COL_INDICES 8
#define NUM_OF_L1_CL_PREFETCHES (L2_MSHRS / CL_SIZE_IN_COL_INDICES)

#ifdef PROFILE_WITH_VTUNE
#include "ittnotify.h"
#endif

#define PREFETCHT0  (3)
#define PREFETCHT2  (2)
#define PREFETCHNTA (0)

static void spmv(uint64_t num_of_rows, const double *vec, const double *mat_vals, const int64_t *pos, const int64_t *crd, double *res)
{

#ifdef PROFILE_WITH_VTUNE
    __itt_resume();
#endif

    for (uint64_t i = 0; i < num_of_rows; i++) {

        const int64_t j_start = pos[i];
        const int64_t j_end = pos[i + 1];

        // Fill l1 mshrs by fetching l1_mshrs cache lines
        int64_t k;
#pragma clang loop unroll_count(CL_SIZE_IN_COL_INDICES)
        for (k = j_start; k < j_start + NUM_OF_L1_CL_PREFETCHES * CL_SIZE_IN_COL_INDICES; k+=CL_SIZE_IN_COL_INDICES) {
            __builtin_prefetch(&crd[k], 0, PREFETCHNTA);
            __builtin_prefetch(&mat_vals[k], 0, PREFETCHNTA);
        }

        // Fill l2 mshrs by fetching l2_mshr elements
        for (int64_t l = j_start; l < j_start + L2_MSHRS; l++) {
            int64_t to_pref = crd[l];
            __builtin_prefetch(&vec[to_pref], 0, PREFETCHT2);
        }

        // Assume that all capacity for MLP is now exhausted:
        // Steady state - start computation

        int64_t num_of_non_zeros_in_row = j_end - j_start;
        int64_t j_end_closest_mult_of_cl_size = j_start + (num_of_non_zeros_in_row / CL_SIZE_IN_COL_INDICES) * CL_SIZE_IN_COL_INDICES;

        double res_i = 0;
        for (int64_t j = j_start; j < j_end_closest_mult_of_cl_size; j+=CL_SIZE_IN_COL_INDICES, k+=CL_SIZE_IN_COL_INDICES) {

#pragma clang loop unroll_count(CL_SIZE_IN_COL_INDICES)
            for (int64_t l = 0; l < CL_SIZE_IN_COL_INDICES; l++) {
                double mat_val = mat_vals[j + l];
                int64_t col_idx = crd[j + l];
                res_i += vec[col_idx] * mat_val;

                int64_t to_pref = crd[j + l + L2_MSHRS];
                __builtin_prefetch(&vec[to_pref], 0, PREFETCHT2);
            }

            __builtin_prefetch(&crd[k], 0, PREFETCHNTA);
            __builtin_prefetch(&mat_vals[k], 0, PREFETCHNTA);
        }

        for (int64_t j = j_end_closest_mult_of_cl_size; j < j_end; j++) {
            res_i += mat_vals[j] * vec[crd[j]];
        }

        res[i] = res_i;
    }

#ifdef PROFILE_WITH_VTUNE
    __itt_detach();
#endif

}

double compute(uint64_t num_of_rows, const double *vec, const double *mat_vals, const int64_t *pos, const int64_t *crd, double *res)
{

    struct timespec start_ts;
    clock_gettime(CLOCK_MONOTONIC, &start_ts);

    spmv(num_of_rows, vec, mat_vals, pos, crd, res);

    struct timespec end_ts;
    clock_gettime(CLOCK_MONOTONIC, &end_ts);

    long seconds = end_ts.tv_sec - start_ts.tv_sec;
    long nanoseconds = end_ts.tv_nsec - start_ts.tv_nsec;
    return seconds + nanoseconds*1e-9;
}
