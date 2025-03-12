#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
module {
  func.func @spmv(%arg0: tensor<10xf64>, %arg1: tensor<10x10xf64, #sparse>, %arg2: tensor<10xf64>) -> tensor<10xf64> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %B = sparse_tensor.values %arg1 : tensor<10x10xf64, #sparse> to memref<?xf64>
    %c = bufferization.to_memref %arg2 : tensor<10xf64> to memref<10xf64>
    %a = bufferization.to_memref %arg0 : tensor<10xf64> to memref<10xf64>
    %Bj_pos = sparse_tensor.positions %arg1 {level = 1 : index} : tensor<10x10xf64, #sparse> to memref<?xindex>
    %Bj_crd = sparse_tensor.coordinates %arg1 {level = 1 : index} : tensor<10x10xf64, #sparse> to memref<?xindex>
    scf.for %i = %c0 to %c10 step %c1 {
      %6 = memref.load %a[%i] : memref<10xf64>
      %7 = memref.load %Bj_pos[%i] : memref<?xindex>
      %8 = arith.addi %i, %c1 : index
      %9 = memref.load %Bj_pos[%8] : memref<?xindex>
      %10 = scf.for %jj = %7 to %9 step %c1 iter_args(%arg5 = %6) -> (f64) {
        %11 = memref.load %Bj_crd[%jj] : memref<?xindex>
        %12 = memref.load %B[%jj] : memref<?xf64>
        %13 = memref.load %c[%11] : memref<10xf64>
        %14 = arith.mulf %12, %13 : f64
        %15 = arith.addf %14, %arg5 : f64
        scf.yield %15 : f64
      } {"Emitted from" = "linalg.generic"}
      memref.store %10, %a[%i] : memref<10xf64>
    } {"Emitted from" = "linalg.generic"}

    scf.for %i = %c0 to %c10 step %c1 {
      %ci = memref.load %c[%i] : memref<10xf64>
      %7 = memref.load %Bj_pos[%i] : memref<?xindex>
      %8 = arith.addi %i, %c1 : index
      %9 = memref.load %Bj_pos[%8] : memref<?xindex>
      scf.for %jj = %7 to %9 step %c1 {
        %j = memref.load %Bj_crd[%jj] : memref<?xindex>
        %aj = memref.load %a[%j] : memref<10xf64>
        %Bij = memref.load %B[%jj] : memref<?xf64>
        %13 = arith.mulf %Bij, %ci : f64
        %14 = arith.addf %13, %aj : f64
        %15 = arith.cmpi eq, %i, %j : index
        %16 = arith.select %15, %14, %aj : f64
        memref.store %16, %a[%j] : memref<10xf64>
      } {"Emitted from" = "linalg.generic"}
    } {"Emitted from" = "linalg.generic"}
    %5 = bufferization.to_tensor %a restrict : memref<10xf64> to tensor<10xf64>

    return %5 : tensor<10xf64>
  }
}
