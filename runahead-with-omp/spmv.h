#ifndef SPMV_RUNAHEAD_WITH_OMP_TASKS_H
#define SPMV_RUNAHEAD_WITH_OMP_TASKS_H

#include <stdint.h>

double compute(uint64_t num_of_rows, const double *vec, const double *mat_vals, const int64_t *pos, const int64_t *crd,
               double *res);

#endif //SPMV_RUNAHEAD_WITH_OMP_TASKS_H
