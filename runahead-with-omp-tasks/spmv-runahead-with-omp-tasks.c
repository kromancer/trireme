#include <omp.h>
#include <stdio.h>

#define min(a, b) ((a) < (b) ? (a) : (b))

#ifndef PD
#error "prefetch distance not defined!"
#endif

static double* a_vals;
static const int64_t* crd;
static const double* B_vals;
static const double* c_vals;

typedef void (*pref_or_comp_task_t)(int start, int end, int row);

typedef enum {
    PREF,
    COMP
} task_type_t;

static void pref_task(int start, int end, int unused) {
    (void) unused;

    for (int j = start; j < end; j++) {
        __builtin_prefetch(&c_vals[crd[j]], 0, 0);
    }
}

static void comp_task(int start, int end, int row) {
    for (int j = start; j < end; j++) {
        a_vals[row] += B_vals[j] * c_vals[crd[j]];
    }
}

static void log_decorator(pref_or_comp_task_t task, int index_start, int index_end, int row, task_type_t task_type) {

    if (index_start > index_end)
        return;

#ifdef ENABLE_LOGS
    double start = omp_get_wtime();
#endif // ENABLE_LOGS

    task(index_start, index_end, row);

#ifdef ENABLE_LOGS
    double end =  omp_get_wtime();

    static int pref_task_id = 0;
    static int comp_task_id = 0;

    int id = -1;
    if (task_type == PREF) {
        id = pref_task_id;
#pragma omp atomic
        pref_task_id++;
    } else {
        id = comp_task_id;
#pragma omp atomic
        comp_task_id++;
    }

#pragma omp critical
    printf("Thread %d %s(%d) start %f s end %f s row %d cols %d - %d\n",
           omp_get_thread_num(),
           task == pref_task ? "pref" : "comp",
           id,
           start,
           end,
           row, index_start, index_end
           );
    fflush(stdout);
#endif // ENABLE_LOGS
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
        for (int i = 0; i < num_of_rows; i++) {

            a_vals[i] = 0.0;
            const int j_start = pos[i];
            const int j_end = pos[i + 1];

            // create pref(0) - no dependencies
#pragma omp task default(firstprivate) depend (out: crd[j_start:PD])
            log_decorator(pref_task, j_start, min(j_start + PD, j_end), 0,
                          PREF);

            // create compute(0) - depends only on pref(0)
#pragma omp task default(firstprivate) \
            depend (out: B_vals[j_start:PD]) \
            depend (in: crd[j_start:PD])
            log_decorator(comp_task, j_start, min(j_start + PD, j_end), i,
                          COMP);

            // create pref(1) - depends only on pref(0)
#pragma omp task default(firstprivate) \
            depend (out: crd[j_start + PD:PD]) \
            depend (in: crd[j_start:PD])
            log_decorator(pref_task, j_start + PD, min(j_start + 2 * PD, j_end), 0,
                          PREF);

            // create compute(1) - depends on compute(0) and pref(1)
#pragma omp task default(firstprivate) \
            depend (out: B_vals[j_start + PD:PD]) \
            depend (in: B_vals[j_start:PD]) \
            depend (in: crd[j_start + PD:PD])
            log_decorator(comp_task, j_start + PD, min(j_start + 2 * PD, j_end), i,
                          COMP);

            for (int j = j_start + 2 * PD; j < j_end; j += PD) {

                // prefetching task
#pragma omp task default(firstprivate) \
                depend (out: crd[j:PD]) \
                depend (in: crd[j-PD:PD])  \
                depend (in: B_vals[j-2*PD:PD])
                log_decorator(pref_task, j, min(j + PD, j_end), i,
                              PREF);

                // compute task
#pragma omp task default(firstprivate) \
                depend (out: B_vals[j:PD]) \
                depend (in: crd[j:PD]) \
                depend (in: crd[j-PD:PD])
                log_decorator(comp_task, j, min(j + PD, j_end), i,
                              COMP);
            }

#ifdef ENABLE_LOGS
            printf("Thread %d: Done making tasks for row %d\n", omp_get_thread_num(), i);
            fflush(stdout);
#endif
        }
    }

    return omp_get_wtime() - start;
}
