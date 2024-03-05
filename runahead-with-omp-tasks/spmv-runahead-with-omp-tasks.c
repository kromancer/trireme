#include "spmv.h"

#include <omp.h>
#include <stdio.h>
#include <stdbool.h>

#define min(a, b) ((a) < (b) ? (a) : (b))

#ifndef PD
#warning "prefetch distance not defined, using 16"
#define PD 16
#endif

#ifndef LOCALITY_HINT
#warning "locality hint (0,1,2 or 3) is not defined, using 0"
#define LOCALITY_HINT 0
#endif

static double* a_vals;
static const int64_t* crd;
static const double* B_vals;
static const double* c_vals;

typedef void (*pref_or_comp_task_t)(int64_t start, int64_t end, int row, int task_pair_counter);

static void comp_task(int64_t start, int64_t end, int row, int unused) {
    (void) unused;

    for (int64_t j = start; j < end; j++) {
        a_vals[row] += B_vals[j] * c_vals[crd[j]];
    }
}

static void log_decorator(pref_or_comp_task_t task, int task_counter, int64_t index_start, int64_t index_end, int row) {

    if (index_start > index_end)
        return;

#ifdef ENABLE_LOGS
    double start = omp_get_wtime();
#endif // ENABLE_LOGS

    task(index_start, index_end, row, task_counter);

#ifdef ENABLE_LOGS
    double end = omp_get_wtime();
#pragma omp critical
    {
        printf("Thread %d on core %d %s(%d) start %f s end %f s row %d B_vals/crd[%lld:%lld]\n",
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
#endif // ENABLE_LOGS
}

static void pref_task(int64_t start, int64_t end, int row, int task_pair_counter) {

    for (int64_t j = start; j < end; j++) {
        __builtin_prefetch(&c_vals[crd[j]], 0, LOCALITY_HINT);
    }

    // compute task
#pragma omp task default(firstprivate) \
                depend (out: B_vals[task_pair_counter]) \
                depend (in: B_vals[task_pair_counter - 1]) if (0)
    log_decorator(comp_task, task_pair_counter, start, end, row);
}

double compute(double* a_vals_, int num_of_rows, const int64_t* pos, const int64_t* crd_,
               const double* B_vals_, const double* c_vals_) {

    printf("omp_get_wtick(): %f s (number of seconds between successive ticks)\n", omp_get_wtick());
    double start = omp_get_wtime();

    a_vals = a_vals_;
    crd = crd_;
    B_vals = B_vals_;
    c_vals = c_vals_;

#pragma omp parallel num_threads(3)
    {
        #pragma omp single
        {

            // prime dependencies
#pragma omp task default(firstprivate) \
                depend (out: crd[0]) \
                depend (out: B_vals[1]) \
                depend (out: B_vals[2]) if (0)
            { /* do nothing */ }

            int task_pair_counter = 2;

            for (int i = 0; i < num_of_rows; i++) {

                a_vals[i] = 0.0;
                const int64_t j_start = pos[i];
                const int64_t j_end = pos[i + 1];

                for (int64_t j = j_start; j < j_end; j += PD) {

                    // prefetching task
#pragma omp task default(firstprivate) \
                depend (out: crd[task_pair_counter]) \
                depend (in: crd[task_pair_counter - 1])  \
                depend (in: B_vals[task_pair_counter - 2])
                    log_decorator(pref_task, task_pair_counter, j, min(j + PD, j_end), i);

                    task_pair_counter++;
                }

#ifdef ENABLE_LOGS
                printf("Thread %d: Done making tasks for row %d\n", omp_get_thread_num(), i);
                fflush(stdout);
#endif
            }
        }
    }

    return omp_get_wtime() - start;
}
