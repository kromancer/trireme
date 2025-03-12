#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
#sparse1 = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
module {
  func.func @spmv(%arg0: tensor<1024xf64>, %arg1: tensor<?xindex>, %arg2: tensor<?xindex>, %arg3: tensor<?xf64>, %arg4: tensor<?xindex>, %arg5: tensor<?xindex>, %arg6: tensor<?xf64>) -> tensor<1024xf64> {
    %0 = sparse_tensor.assemble (%arg1, %arg2), %arg3 : (tensor<?xindex>, tensor<?xindex>), tensor<?xf64> to tensor<1024x1024xf64, #sparse>
    %1 = sparse_tensor.assemble (%arg4, %arg5), %arg6 : (tensor<?xindex>, tensor<?xindex>), tensor<?xf64> to tensor<1024xf64, #sparse1>
    %2 = call @_internal_spmv(%arg0, %0, %1) : (tensor<1024xf64>, tensor<1024x1024xf64, #sparse>, tensor<1024xf64, #sparse1>) -> tensor<1024xf64>
    return %2 : tensor<1024xf64>
  }
  func.func private @_internal_spmv(%arg0: tensor<1024xf64>, %arg1: tensor<1024x1024xf64, #sparse>, %arg2: tensor<1024xf64, #sparse1>) -> tensor<1024xf64> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %0 = sparse_tensor.values %arg1 : tensor<1024x1024xf64, #sparse> to memref<?xf64>
    %1 = sparse_tensor.values %arg2 : tensor<1024xf64, #sparse1> to memref<?xf64>
    %2 = bufferization.to_memref %arg0 : memref<1024xf64>
    %Bi_pos = sparse_tensor.positions %arg1 {level = 1 : index} : tensor<1024x1024xf64, #sparse> to memref<?xindex>
    %4 = sparse_tensor.coordinates %arg1 {level = 1 : index} : tensor<1024x1024xf64, #sparse> to memref<?xindex>
    %5 = sparse_tensor.positions %arg2 {level = 0 : index} : tensor<1024xf64, #sparse1> to memref<?xindex>
    %6 = sparse_tensor.coordinates %arg2 {level = 0 : index} : tensor<1024xf64, #sparse1> to memref<?xindex>


    scf.parallel (%i) = (%c0) to (%c1024) step (%c1) {
      %a_i = memref.load %2[%i] : memref<1024xf64>

      // Bi_pos[i] and Bi_pos[i+1] and
      %Bi_pos_i = memref.load %Bi_pos[%i] : memref<?xindex>
      %10 = arith.addi %i, %c1 : index
      %Bi_pos_i_p_1 = memref.load %Bi_pos[%10] : memref<?xindex>

      // c_pos[0] and c_pos[1]
      %12 = memref.load %5[%c0] : memref<?xindex>
      %13 = memref.load %5[%c1] : memref<?xindex>

      // while (ii < Bi_pos[i+1] && kk < c_pos[1])
      %14:3 = scf.while (%ii = %Bi_pos_i, %kk = %12, %acc_old = %a_i) : (index, index, f64) -> (index, index, f64) {
        %15 = arith.cmpi ult, %ii, %Bi_pos_i_p_1 : index
        %16 = arith.cmpi ult, %kk, %13 : index
        %17 = arith.andi %15, %16 : i1
        scf.condition(%17) %ii, %kk, %acc_old : index, index, f64
      } do {
      ^bb0(%ii: index, %kk: index, %acc_old: f64):
        // j = min(jc, jB)
        %jB = memref.load %4[%ii] : memref<?xindex>
        %jc = memref.load %6[%kk] : memref<?xindex>
        %17 = arith.cmpi ult, %jc, %jB : index
        %j = arith.select %17, %jc, %jB : index

        // if (jc == j && jB == j)
        // multiply and accumulate
        %19 = arith.cmpi eq, %jB, %j : index
        %20 = arith.cmpi eq, %jc, %j : index
        %21 = arith.andi %19, %20 : i1
        %22 = scf.if %21 -> (f64) {
          %29 = memref.load %0[%ii] : memref<?xf64>
          %30 = memref.load %1[%kk] : memref<?xf64>
          %31 = arith.mulf %29, %30 : f64
          %32 = arith.addf %acc_old, %31 : f64
          scf.yield %32 : f64
        } else {
          scf.yield %acc_old : f64
        }

        // ii += (jB == j)
        %23 = arith.cmpi eq, %jB, %j : index
        %24 = arith.addi %ii, %c1 : index
        %25 = arith.select %23, %24, %ii : index

        // kk += (jc == j)
        %26 = arith.cmpi eq, %jc, %j : index
        %27 = arith.addi %kk, %c1 : index
        %28 = arith.select %26, %27, %kk : index

        scf.yield %25, %28, %22 : index, index, f64
      } attributes {"Emitted from" = "linalg.generic"}
      memref.store %14#2, %2[%i] : memref<1024xf64>
      scf.reduce
    } {"Emitted from" = "linalg.generic"}
    %7 = bufferization.to_tensor %2 : memref<1024xf64>
    return %7 : tensor<1024xf64>
  }
}
