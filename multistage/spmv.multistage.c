#define _GNU_SOURCE

#include "spmv.h"

#include <time.h>

#if defined (DISABLE_HW_PREF_L1_IPP) || defined (DISABLE_HW_PREF_L1_NPP)
#include <assert.h>
#include <sched.h>
#include "msr.h"
#endif

#define CL_SIZE_IN_COL_INDICES 8

#ifndef L1_MSHRS
#warning "L1_MSHRS not defined, using 10"
#define L1_MSHRS 10
#endif

#ifndef L2_MSHRS
#warning "L2_MSHRS not defined, using 40"
#define L2_MSHRS 40
#endif

#ifndef DISABLE_HW_PREF_L1_IPP
#warning "DISBLE_HW_PREF_L1_IPP not defined, L1 IPP will be ENABLED (applicapable for only on intel atom cores)"
#define DISABLE_HW_PREF_L1_IPP 0
#endif

#ifndef DISABLE_HW_PREF_L1_NPP
#warning "DISBLE_HW_PREF_L1_NPP not defined, L1 NPP will be ENABLED (applicapable for only on intel atom cores)"
#define DISABLE_HW_PREF_L1_NPP 0
#endif


#define PREFETCHT2  2
#define PREFETCHT0  3


double compute(uint64_t num_of_rows, const double *vec, const double *mat_vals, const int64_t *pos, const int64_t *crd, double *res) {

#if DISABLE_HW_PREF_L1_IPP == 1 || DISABLE_HW_PREF_L1_NPP == 1
    union msr_u hwpf_msr_value[HWPF_MSR_FIELDS];
    int core_id = sched_getcpu();
    int msr_file = msr_int(core_id, hwpf_msr_value);
#endif

#if DISABLE_HW_PREF_L1_IPP == 1
    assert(msr_disable_l1ipp(hwpf_msr_value) == 0);
#endif

#if DISABLE_HW_PREF_L1_NPP == 1
    assert(msr_disable_l1npp(hwpf_msr_value) == 0);
#endif

    struct timespec start_ts;
    clock_gettime(CLOCK_MONOTONIC, &start_ts);


    for (uint64_t i = 0; i < num_of_rows; i++) {

        res[i] = 0.0;
        const int64_t j_start = pos[i];
        const int64_t j_end = pos[i + 1];

        // Fill l1 mshrs by fetching l1_mshrs cache lines
        int64_t k;
#pragma clang loop unroll_count(CL_SIZE_IN_COL_INDICES)
        for (k = j_start; k < j_start + L1_MSHRS * CL_SIZE_IN_COL_INDICES; k+=CL_SIZE_IN_COL_INDICES) {
            __builtin_prefetch(&crd[k], 0, PREFETCHT0);
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

            __builtin_prefetch(&crd[k], 0, PREFETCHT0);
        }

        for (int64_t j = j_end_closest_mult_of_cl_size; j < j_end; j++) {
            res[i] += mat_vals[j] * vec[crd[j]];
        }
    }

    struct timespec end_ts;
    clock_gettime(CLOCK_MONOTONIC, &end_ts);

#if DISABLE_HW_PREF_L1_IPP == 1
    assert(msr_enable_l1ipp(hwpf_msr_value) == 1);
#endif

#if DISABLE_HW_PREF_L1_NPP == 1
    assert(msr_enable_l1npp(hwpf_msr_value) == 1);
#endif

    long seconds = end_ts.tv_sec - start_ts.tv_sec;
    long nanoseconds = end_ts.tv_nsec - start_ts.tv_nsec;
    return seconds + nanoseconds*1e-9;
}
