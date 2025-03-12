







void spmv(int N, double *restrict a, double *restrict c, int *restrict Bj_pos,
          int *restrict Bj_crd, double *restrict B) {



    for (int i = 0; i < N; i++) {

        int seg_start = Bj_pos[i];
        int seg_end = Bj_pos[i+1];

        for (int jj = seg_start; jj < seg_end; jj++) {
            int j = Bj_crd[jj];
            a[i] += B[jj] * c[j];
            a[j] += B[jj] * c[i];
        }

    }


}
