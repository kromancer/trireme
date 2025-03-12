#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton(soa)) }>
module {
  func.func private @spmv(%arg0: tensor<1024x1024xf64, #sparse>, %arg1: tensor<1024xf64>, %arg2: tensor<1024xf64>) -> tensor<1024xf64> {
    %c32 = arith.constant 32 : index
    %true = arith.constant true
    %false = arith.constant false
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = sparse_tensor.values %arg0 : tensor<1024x1024xf64, #sparse> to memref<?xf64>
    %1 = bufferization.to_memref %arg1 : tensor<1024xf64> to memref<1024xf64>
    %2 = bufferization.to_memref %arg2 : tensor<1024xf64> to memref<1024xf64>
    %3 = sparse_tensor.positions %arg0 {level = 0 : index} : tensor<1024x1024xf64, #sparse> to memref<?xindex>
    %4 = sparse_tensor.coordinates %arg0 {level = 0 : index} : tensor<1024x1024xf64, #sparse> to memref<?xindex>
    %5 = sparse_tensor.coordinates %arg0 {level = 1 : index} : tensor<1024x1024xf64, #sparse> to memref<?xindex>
    %6 = memref.load %3[%c0] : memref<?xindex>
    %7 = memref.load %3[%c1] : memref<?xindex>
    %8 = scf.while (%arg3 = %6) : (index) -> index {
      %11 = arith.cmpi ult, %arg3, %7 : index
      %12 = scf.if %11 -> (i1) {
        %13 = memref.load %4[%6] : memref<?xindex>
        %14 = memref.load %4[%arg3] : memref<?xindex>
        %15 = arith.cmpi eq, %13, %14 : index
        scf.yield %15 : i1
      } else {
        scf.yield %false : i1
      }
      scf.condition(%12) %arg3 : index
    } do {
    ^bb0(%arg3: index):
      %11 = arith.addi %arg3, %c1 : index
      scf.yield %11 : index
    }
    %9:2 = scf.while (%arg3 = %6, %arg4 = %8) : (index, index) -> (index, index) {
      %11 = arith.cmpi ult, %arg3, %7 : index
      scf.condition(%11) %arg3, %arg4 : index, index
    } do {
    ^bb0(%arg3: index, %arg4: index):
      %11 = memref.load %4[%arg3] : memref<?xindex>
      %12 = arith.addi %arg3, %c1 : index
      %13 = arith.cmpi ult, %12, %7 : index
      %14 = arith.select %13, %12, %7 : index
      %15 = memref.load %4[%14] : memref<?xindex>
      memref.prefetch %2[%15], write, locality<2>, data : memref<1024xf64>
      scf.if %true {
        %17 = memref.load %2[%11] : memref<1024xf64>
        %18 = scf.for %arg5 = %arg3 to %arg4 step %c1 iter_args(%arg6 = %17) -> (f64) {
          %19 = memref.load %5[%arg5] : memref<?xindex>
          %20 = arith.addi %arg5, %c32 : index
          %21 = arith.cmpi ult, %20, %7 : index
          %22 = arith.select %21, %20, %7 : index
          %23 = memref.load %5[%22] : memref<?xindex>
          memref.prefetch %1[%23], read, locality<2>, data : memref<1024xf64>
          %24 = memref.load %0[%arg5] : memref<?xf64>
          %25 = memref.load %1[%19] : memref<1024xf64>
          %26 = arith.mulf %24, %25 : f64
          %27 = arith.addf %arg6, %26 : f64
          scf.yield %27 : f64
        } {"Emitted from" = "linalg.generic"}
        memref.store %18, %2[%11] : memref<1024xf64>
      } else {
      }
      %16:2 = scf.if %true -> (index, index) {
        %17 = scf.while (%arg5 = %arg4) : (index) -> index {
          %18 = arith.cmpi ult, %arg5, %7 : index
          %19 = scf.if %18 -> (i1) {
            %20 = memref.load %4[%arg4] : memref<?xindex>
            %21 = memref.load %4[%arg5] : memref<?xindex>
            %22 = arith.cmpi eq, %20, %21 : index
            scf.yield %22 : i1
          } else {
            scf.yield %false : i1
          }
          scf.condition(%19) %arg5 : index
        } do {
        ^bb0(%arg5: index):
          %18 = arith.addi %arg5, %c1 : index
          scf.yield %18 : index
        }
        scf.yield %arg4, %17 : index, index
      } else {
        scf.yield %arg3, %arg4 : index, index
      }
      scf.yield %16#0, %16#1 : index, index
    } attributes {"Emitted from" = "linalg.generic"}
    %10 = bufferization.to_tensor %2 restrict : memref<1024xf64> to tensor<1024xf64>
    return %10 : tensor<1024xf64>
  }
}
