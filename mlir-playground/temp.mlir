#map = affine_map<(d0) -> (d0, d0)>
#map1 = affine_map<(d0) -> (d0)>
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
module {
  func.func @spmv(%arg0: tensor<10xf64>, %arg1: tensor<10x10xf64, #sparse>, %arg2: tensor<10xf64>) -> tensor<10xf64> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %0 = sparse_tensor.values %arg1 : tensor<10x10xf64, #sparse> to memref<?xf64>
    %1 = bufferization.to_memref %arg2 : tensor<10xf64> to memref<10xf64>
    %2 = bufferization.to_memref %arg0 : tensor<10xf64> to memref<10xf64>
    %3 = sparse_tensor.positions %arg1 {level = 1 : index} : tensor<10x10xf64, #sparse> to memref<?xindex>
    %4 = sparse_tensor.coordinates %arg1 {level = 1 : index} : tensor<10x10xf64, #sparse> to memref<?xindex>
    scf.for %arg3 = %c0 to %c10 step %c1 {
      %7 = memref.load %2[%arg3] : memref<10xf64>
      %8 = memref.load %3[%arg3] : memref<?xindex>
      %9 = arith.addi %arg3, %c1 : index
      %10 = memref.load %3[%9] : memref<?xindex>
      %11 = scf.for %arg4 = %8 to %10 step %c1 iter_args(%arg5 = %7) -> (f64) {
        %12 = memref.load %4[%arg4] : memref<?xindex>
        %13 = memref.load %0[%arg4] : memref<?xf64>
        %14 = memref.load %1[%12] : memref<10xf64>
        %15 = arith.mulf %13, %14 : f64
        %16 = arith.addf %15, %arg5 : f64
        scf.yield %16 : f64
      } {"Emitted from" = "linalg.generic"}
      memref.store %11, %2[%arg3] : memref<10xf64>
    } {"Emitted from" = "linalg.generic"}
    %5 = bufferization.to_tensor %2 restrict : memref<10xf64> to tensor<10xf64>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel"]} ins(%arg1, %arg2 : tensor<10x10xf64, #sparse>, tensor<10xf64>) outs(%5 : tensor<10xf64>) attrs =  {sorted = true} {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %7 = arith.mulf %in, %in_0 : f64
      %8 = arith.subf %7, %out : f64
      linalg.yield %8 : f64
    } -> tensor<10xf64>
    return %6 : tensor<10xf64>
  }
}

