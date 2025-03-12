#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
module {
  func.func @mat_add(%arg0: tensor<6x6xf64>, %arg1: tensor<6x6xf64, #sparse>, %arg2: tensor<6x6xf64, #sparse>) -> tensor<6x6xf64> {
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_memref %arg0 : memref<6x6xf64>
    %1 = sparse_tensor.values %arg1 : tensor<6x6xf64, #sparse> to memref<?xf64>
    %2 = bufferization.to_memref %arg0 : memref<6x6xf64>
    linalg.fill ins(%cst : f64) outs(%2 : memref<6x6xf64>)
    %3 = sparse_tensor.positions %arg1 {level = 1 : index} : tensor<6x6xf64, #sparse> to memref<?xindex>
    %4 = sparse_tensor.coordinates %arg1 {level = 1 : index} : tensor<6x6xf64, #sparse> to memref<?xindex>
    scf.for %arg3 = %c0 to %c6 step %c1 {
      %10 = memref.load %3[%arg3] : memref<?xindex>
      %11 = arith.addi %arg3, %c1 : index
      %12 = memref.load %3[%11] : memref<?xindex>
      %13:2 = scf.while (%arg4 = %10, %arg5 = %c0) : (index, index) -> (index, index) {
        %14 = arith.cmpi ult, %arg4, %12 : index
        scf.condition(%14) %arg4, %arg5 : index, index
      } do {
      ^bb0(%arg4: index, %arg5: index):
        %14 = memref.load %4[%arg4] : memref<?xindex>
        %15 = arith.cmpi eq, %14, %arg5 : index
        scf.if %15 {
          %20 = memref.load %0[%arg3, %arg5] : memref<6x6xf64>
          %21 = memref.load %1[%arg4] : memref<?xf64>
          %22 = arith.addf %20, %21 : f64
          memref.store %22, %2[%arg3, %arg5] : memref<6x6xf64>
        } else {
          scf.if %true {
            %20 = memref.load %0[%arg3, %arg5] : memref<6x6xf64>
            memref.store %20, %2[%arg3, %arg5] : memref<6x6xf64>
          } else {
          }
        }
        %16 = arith.cmpi eq, %14, %arg5 : index
        %17 = arith.addi %arg4, %c1 : index
        %18 = arith.select %16, %17, %arg4 : index
        %19 = arith.addi %arg5, %c1 : index
        scf.yield %18, %19 : index, index
      } attributes {"Emitted from" = "linalg.generic"}
      scf.for %arg4 = %13#1 to %c6 step %c1 {
        %14 = memref.load %0[%arg3, %arg4] : memref<6x6xf64>
        memref.store %14, %2[%arg3, %arg4] : memref<6x6xf64>
      } {"Emitted from" = "linalg.generic"}
    } {"Emitted from" = "linalg.generic"}
    %5 = sparse_tensor.values %arg2 : tensor<6x6xf64, #sparse> to memref<?xf64>
    %6 = bufferization.to_memref %arg0 : memref<6x6xf64>
    linalg.fill ins(%cst : f64) outs(%6 : memref<6x6xf64>)
    %7 = sparse_tensor.positions %arg2 {level = 1 : index} : tensor<6x6xf64, #sparse> to memref<?xindex>
    %8 = sparse_tensor.coordinates %arg2 {level = 1 : index} : tensor<6x6xf64, #sparse> to memref<?xindex>
    scf.for %arg3 = %c0 to %c6 step %c1 {
      %10 = memref.load %7[%arg3] : memref<?xindex>
      %11 = arith.addi %arg3, %c1 : index
      %12 = memref.load %7[%11] : memref<?xindex>
      %13:2 = scf.while (%arg4 = %10, %arg5 = %c0) : (index, index) -> (index, index) {
        %14 = arith.cmpi ult, %arg4, %12 : index
        scf.condition(%14) %arg4, %arg5 : index, index
      } do {
      ^bb0(%arg4: index, %arg5: index):
        %14 = memref.load %8[%arg4] : memref<?xindex>
        %15 = arith.cmpi eq, %14, %arg5 : index
        scf.if %15 {
          %20 = memref.load %2[%arg3, %arg5] : memref<6x6xf64>
          %21 = memref.load %5[%arg4] : memref<?xf64>
          %22 = arith.addf %20, %21 : f64
          memref.store %22, %6[%arg3, %arg5] : memref<6x6xf64>
        } else {
          scf.if %true {
            %20 = memref.load %2[%arg3, %arg5] : memref<6x6xf64>
            memref.store %20, %6[%arg3, %arg5] : memref<6x6xf64>
          } else {
          }
        }
        %16 = arith.cmpi eq, %14, %arg5 : index
        %17 = arith.addi %arg4, %c1 : index
        %18 = arith.select %16, %17, %arg4 : index
        %19 = arith.addi %arg5, %c1 : index
        scf.yield %18, %19 : index, index
      } attributes {"Emitted from" = "linalg.generic"}
      scf.for %arg4 = %13#1 to %c6 step %c1 {
        %14 = memref.load %2[%arg3, %arg4] : memref<6x6xf64>
        memref.store %14, %6[%arg3, %arg4] : memref<6x6xf64>
      } {"Emitted from" = "linalg.generic"}
    } {"Emitted from" = "linalg.generic"}
    %9 = bufferization.to_tensor %6 : memref<6x6xf64>
    return %9 : tensor<6x6xf64>
  }
}

