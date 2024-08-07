// Generated by the Tensor Algebra Compiler (tensor-compiler.org)
// taco "a(i) = B(i,j) * c(j)" -f=B:uq

// Modified to remove references to `taco_tensor_t` by passing everything as func args
// Removed 0 initialization of a

int compute(double* restrict a_vals,
            int* restrict B1_pos,
            int* restrict B1_crd,
            int* restrict B2_crd,
            double* restrict B_vals,
            double* restrict c_vals) {

  int iB = B1_pos[0];
  int pB1_end = B1_pos[1];

  while (iB < pB1_end) {
    int i = B1_crd[iB];
    int B1_segend = iB + 1;
    while (B1_segend < pB1_end && B1_crd[B1_segend] == i) {
      B1_segend++;
    }
    double tja_val = 0.0;
    for (int jB = iB; jB < B1_segend; jB++) {
      int j = B2_crd[jB];
      tja_val += B_vals[jB] * c_vals[j];
    }
    a_vals[i] = tja_val;
    iB = B1_segend;
  }
  return 0;
}
