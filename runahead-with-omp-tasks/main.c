#include <assert.h>
#include <stdio.h>

#include "spmv.h"

#define NUM_OF_ROWS 1
#define NUM_OF_COLS 5

int main() {

    double a_vals[NUM_OF_ROWS] = {0};

    int64_t pos[NUM_OF_ROWS + 1] = {0, 5};
    int64_t crd[NUM_OF_COLS] = {0, 1, 2, 3, 4};
    double B_vals[NUM_OF_COLS] = {3, 2, 4, 5, 0.5};

    double c_vals[NUM_OF_COLS] = {2, 5, 0.5, 4, 2};

    double elapsed_wtime = compute(a_vals, NUM_OF_ROWS, pos, crd, B_vals, c_vals);
    printf("Elapsed wtime: %f s\n", elapsed_wtime);

    const double expected_a_vals[NUM_OF_ROWS] = { 39 };
    for (int i = 0; i < NUM_OF_ROWS; i++) {
        printf("a_vals[%d]: %f\n", i, a_vals[i]);
        assert(a_vals[i] == expected_a_vals[i]);
    }

    return 0;
}
