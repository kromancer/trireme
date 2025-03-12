



#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
#sparse1 = #sparse_tensor.encoding<{ map = (d0, d1) -> (d1 : dense, d0 : compressed) }>

module {
  func.func @spmv(%arg0: tensor<160000xf64>, %arg1: tensor<?xindex>, %arg2: tensor<?xindex>, %arg3: tensor<?xf64>, %arg4: tensor<160000xf64>) -> tensor<160000xf64> {
    %0 = sparse_tensor.assemble (%arg1, %arg2), %arg3 : (tensor<?xindex>, tensor<?xindex>), tensor<?xf64> to tensor<160000x160000xf64, #sparse>
    %1 = sparse_tensor.reinterpret_map %0 : tensor<160000x160000xf64, #sparse> to tensor<160000x160000xf64, #sparse1>
    %2 = call @_internal_spmv(%arg0, %1, %arg4) : (tensor<160000xf64>, tensor<160000x160000xf64, #sparse1>, tensor<160000xf64>) -> tensor<160000xf64>
    return %2 : tensor<160000xf64>
  }
  func.func private @_internal_spmv(%arg0: tensor<160000xf64>, %arg1: tensor<160000x160000xf64, #sparse1>, %arg2: tensor<160000xf64>) -> tensor<160000xf64> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c160000 = arith.constant 160000 : index
    %0 = sparse_tensor.reinterpret_map %arg1 : tensor<160000x160000xf64, #sparse1> to tensor<160000x160000xf64, #sparse>
    %1 = sparse_tensor.values %0 : tensor<160000x160000xf64, #sparse> to memref<?xf64>
    %2 = bufferization.to_memref %arg2 : tensor<160000xf64> to memref<160000xf64>
    %3 = bufferization.to_memref %arg0 : tensor<160000xf64> to memref<160000xf64>
    %4 = sparse_tensor.positions %0 {level = 1 : index} : tensor<160000x160000xf64, #sparse> to memref<?xindex>
    %5 = sparse_tensor.coordinates %0 {level = 1 : index} : tensor<160000x160000xf64, #sparse> to memref<?xindex>
    scf.parallel (%arg3) = (%c0) to (%c160000) step (%c1) {
      %7 = memref.load %2[%arg3] : memref<160000xf64>
      %8 = memref.load %4[%arg3] : memref<?xindex>
      %9 = arith.addi %arg3, %c1 : index
      %10 = memref.load %4[%9] : memref<?xindex>
      scf.for %arg4 = %8 to %10 step %c1 {
        %11 = memref.load %5[%arg4] : memref<?xindex>
        %12 = memref.load %3[%11] : memref<160000xf64>
        %13 = memref.load %1[%arg4] : memref<?xf64>
        %14 = arith.mulf %13, %7 : f64
        %15 = arith.addf %12, %14 : f64
        memref.store %15, %3[%11] : memref<160000xf64>
      } {"Emitted from" = "linalg.generic"}
      scf.reduce
    } {"Emitted from" = "linalg.generic"}
    %6 = bufferization.to_tensor %3 restrict : memref<160000xf64> to tensor<160000xf64>
    return %6 : tensor<160000xf64>
  }
}
