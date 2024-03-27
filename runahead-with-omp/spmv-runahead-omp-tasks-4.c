#include "spmv.h"

#include <omp.h>

#ifdef ENABLE_LOGS
#include <stdio.h>
#endif

#define min(a, b) ((a) < (b) ? (a) : (b))

#ifndef PD
#warning "prefetch distance not defined, using 16"
#define PD 16
#endif

#define LFB_SIZE_FOR_L1D 10

#define NUM_OF_DOUBLES_PER_CL 8

#define PREFETCHT2  2
#define PREFETCHT0  3

static double* res;
static const int64_t* crd;
static const double* mat_vals;
static const double* vec;

typedef void (*pref_or_comp_task_t)(int64_t start, int64_t end, uint64_t row);

static void comp_task(int64_t start, int64_t end, uint64_t row) {
    for (int64_t j = start; j < end; j++) {
        res[row] += mat_vals[j] * vec[crd[j]];
    }
}

static void pref_task(int64_t start, int64_t end, uint64_t unused) {

    (void) unused;

    // Fill L1D LFBs
    for (int64_t j = start; j < LFB_SIZE_FOR_L1D * NUM_OF_DOUBLES_PER_CL; j+=NUM_OF_DOUBLES_PER_CL) {
        __builtin_prefetch(&crd[j], 0, PREFETCHT0);
    }

    for (int64_t j = start; j < end; j+=NUM_OF_DOUBLES_PER_CL) {
        __builtin_prefetch(&vec[crd[j]], 0, PREFETCHT2);
        __builtin_prefetch(&crd[j + LFB_SIZE_FOR_L1D], 0, PREFETCHT0);
    }
}

static void log_decorator(pref_or_comp_task_t task, int task_counter, int64_t index_start, int64_t index_end, uint64_t row) {

#ifdef ENABLE_LOGS
    double start = omp_get_wtime();
#endif // ENABLE_LOGS

    task(index_start, index_end, row);

#ifdef ENABLE_LOGS
    double end = omp_get_wtime();
#pragma omp critical
    {
        printf("Thread %d on core %d %s(%d) start %f s end %f s row %lu mat_vals/crd[%ld:%ld]\n",
               omp_get_thread_num(),
               omp_get_place_num(),
               task == comp_task ? "comp" : "pref",
               task_counter,
               start,
               end,
               row, index_start, index_end
        );
        fflush(stdout);
    }
#else
    (void) task_counter;
#endif // ENABLE_LOGS
}

double
compute(uint64_t num_of_rows, const double *vec_, const double *mat_vals_, const int64_t *pos, const int64_t *crd_,
        double *res_) {

    double start = omp_get_wtime();

    res = res_;
    crd = crd_;
    vec = vec_;
    mat_vals = mat_vals_;

#pragma omp parallel num_threads(2)
    {
#pragma omp single
        {
            int task_pair_counter = 0;

            for (uint64_t i = 0; i < num_of_rows; i++) {

                res[i] = 0.0;
                const int64_t j_start = pos[i];
                const int64_t j_end = pos[i + 1];

                for (int64_t j = j_start; j < j_end; j += PD) {

#pragma omp task firstprivate(j, j_end, i, task_pair_counter) mergeable if (task_pair_counter % 2 == 1)
                    log_decorator(pref_task, task_pair_counter, j, min(j + PD, j_end), i);

#pragma omp task firstprivate(j, j_end, i, task_pair_counter) depend (out: res) mergeable if (task_pair_counter % 2 == 1)
                    log_decorator(comp_task, task_pair_counter, j, min(j + PD, j_end), i);

                    task_pair_counter++;
                }

#ifdef ENABLE_LOGS
                printf("Thread %d: Done making tasks for row %lu\n", omp_get_thread_num(), i);
                fflush(stdout);
#endif
            }
        }
    }

    return omp_get_wtime() - start;
}
