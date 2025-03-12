#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0: dense, d1: compressed) }>

// Computes: a(i) = B(i,j) * c(j)
func.func @spmv(%a_t: tensor<1024xf64>,
                %B: tensor<1024x1024xf64, #sparse>,
                %c_t: tensor<1024xf64>) -> tensor<1024xf64> {

    %c3 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %B2_pos = sparse_tensor.positions %B {level = 1 : index} : tensor<1024x1024xf64, #sparse> to memref<?xindex>
    %B2_crd = sparse_tensor.coordinates %B {level = 1 : index} : tensor<1024x1024xf64, #sparse> to memref<?xindex>
    %B_vals = sparse_tensor.values %B : tensor<1024x1024xf64, #sparse> to memref<?xf64>

    %c_vals = bufferization.to_memref %c_t : memref<1024xf64>
    %a_buff = bufferization.to_memref %a_t : memref<1024xf64>

    // Fixed lookahead distance
    %dist = arith.constant 42 : index
    %double_dist = arith.muli %dist, %c2 : index

    scf.for %i = %c0 to %c3 step %c1 {
        %6 = memref.load %a_buff[%i] : memref<1024xf64>

        // Load position of B2 segment start
        %pB2 = memref.load %B2_pos[%i] : memref<?xindex>

        // Load position of B2 segment end
        %i_plus_one = arith.addi %i, %c1 : index
        %pB2_end = memref.load %B2_pos[%i_plus_one] : memref<?xindex>

        %a_i = scf.for %jB = %pB2 to %pB2_end step %c1 iter_args(%arg5 = %6) -> (f64) {

            // Load B_vals[jB]
            %B_vals_jB = memref.load %B_vals[%jB] : memref<?xf64>

            // Load candidate coordinate j
            %j = memref.load %B2_crd[%jB] : memref<?xindex>

            // Load c_vals[j]
            %c_vals_j = memref.load %c_vals[%j] : memref<1024xf64>

            // Prefetch B2_crd[jB + 2 * dist]
            %jB_plus_double_dist = arith.addi %jB, %double_dist : index
            memref.prefetch %B2_crd[%jB_plus_double_dist], read, locality<3>, data : memref<?xindex>

            // Load B2_crd[min(j + dist, j_end)]
            %jB_plus_dist = arith.addi %jB, %dist : index
            %jB_plus_dist_or_pB2_end = arith.minui %jB_plus_dist, %pB2_end : index
            %B2_crd_jB_plus_dist_or_pB2_end = memref.load %B2_crd[%jB_plus_dist_or_pB2_end] : memref<?xindex>

            // Prefetch c_vals[B2_crd[%B2_crd_jB_plus_dist_or_pB2_end]]
            memref.prefetch %c_vals[%B2_crd_jB_plus_dist_or_pB2_end], read, locality<3>, data : memref<1024xf64>


            %mul = arith.mulf %B_vals_jB, %c_vals_j : f64
            %a_i = arith.addf %arg5, %mul : f64
            scf.yield %a_i : f64
        }

        memref.store %a_i, %a_buff[%i] : memref<1024xf64>
    }

    %5 = bufferization.to_tensor %a_buff restrict : memref<1024xf64>
    return %5 : tensor<1024xf64>
}
