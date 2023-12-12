extern void some_function(int node);

void compute(int *restrict key_buff1, int *restrict key_buff2, int num_keys) {
  for (int i = 0; i < num_keys; i++) {
    some_function(key_buff1[key_buff2[i]]);
  }
}
