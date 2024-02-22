#ifndef SPMV_RUNAHEAD_WITH_OMP_TASKS_H
#define SPMV_RUNAHEAD_WITH_OMP_TASKS_H

#include <stdint.h>

double compute(double* A_vals, int num_of_rows, const int64_t* pos, const int64_t* crd,
               const double* B_vals, const double* c_vals);

#endif //SPMV_RUNAHEAD_WITH_OMP_TASKS_H
