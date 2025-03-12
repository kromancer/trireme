#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
module {
  func.func @mat_add(%arg0: tensor<6x6xf64>, %arg1: tensor<6x6xf64, #sparse>, %arg2: tensor<6x6xf64, #sparse>) -> tensor<6x6xf64> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = sparse_tensor.values %arg1 : tensor<6x6xf64, #sparse> to memref<?xf64>
    %1 = sparse_tensor.values %arg2 : tensor<6x6xf64, #sparse> to memref<?xf64>
    %2 = bufferization.to_memref %arg0 : memref<6x6xf64>
    linalg.fill ins(%cst : f64) outs(%2 : memref<6x6xf64>)
    %3 = sparse_tensor.positions %arg1 {level = 1 : index} : tensor<6x6xf64, #sparse> to memref<?xindex>
    %4 = sparse_tensor.coordinates %arg1 {level = 1 : index} : tensor<6x6xf64, #sparse> to memref<?xindex>
    %5 = sparse_tensor.positions %arg2 {level = 1 : index} : tensor<6x6xf64, #sparse> to memref<?xindex>
    %6 = sparse_tensor.coordinates %arg2 {level = 1 : index} : tensor<6x6xf64, #sparse> to memref<?xindex>
    scf.for %arg3 = %c0 to %c6 step %c1 {
      %8 = memref.load %3[%arg3] : memref<?xindex>
      %9 = arith.addi %arg3, %c1 : index
      %10 = memref.load %3[%9] : memref<?xindex>
      %11 = memref.load %5[%arg3] : memref<?xindex>
      %12 = arith.addi %arg3, %c1 : index
      %13 = memref.load %5[%12] : memref<?xindex>
      %14:2 = scf.while (%arg4 = %8, %arg5 = %11) : (index, index) -> (index, index) {
        %15 = arith.cmpi ult, %arg4, %10 : index
        %16 = arith.cmpi ult, %arg5, %13 : index
        %17 = arith.andi %15, %16 : i1
        scf.condition(%17) %arg4, %arg5 : index, index
      } do {
      ^bb0(%arg4: index, %arg5: index):
        %15 = memref.load %4[%arg4] : memref<?xindex>
        %16 = memref.load %6[%arg5] : memref<?xindex>
        %17 = arith.cmpi ult, %16, %15 : index
        %18 = arith.select %17, %16, %15 : index
        %19 = arith.cmpi eq, %15, %18 : index
        %20 = arith.cmpi eq, %16, %18 : index
        %21 = arith.andi %19, %20 : i1
        scf.if %21 {
          %28 = memref.load %0[%arg4] : memref<?xf64>
          %29 = memref.load %1[%arg5] : memref<?xf64>
          %30 = arith.addf %28, %29 : f64
          memref.store %30, %2[%arg3, %18] : memref<6x6xf64>
        } else {
          %28 = arith.cmpi eq, %15, %18 : index
          scf.if %28 {
            %29 = memref.load %0[%arg4] : memref<?xf64>
            memref.store %29, %2[%arg3, %18] : memref<6x6xf64>
          } else {
            %29 = arith.cmpi eq, %16, %18 : index
            scf.if %29 {
              %30 = memref.load %1[%arg5] : memref<?xf64>
              memref.store %30, %2[%arg3, %18] : memref<6x6xf64>
            } else {
            }
          }
        }
        %22 = arith.cmpi eq, %15, %18 : index
        %23 = arith.addi %arg4, %c1 : index
        %24 = arith.select %22, %23, %arg4 : index
        %25 = arith.cmpi eq, %16, %18 : index
        %26 = arith.addi %arg5, %c1 : index
        %27 = arith.select %25, %26, %arg5 : index
        scf.yield %24, %27 : index, index
      } attributes {"Emitted from" = "linalg.generic"}
      scf.for %arg4 = %14#0 to %10 step %c1 {
        %15 = memref.load %4[%arg4] : memref<?xindex>
        %16 = memref.load %0[%arg4] : memref<?xf64>
        memref.store %16, %2[%arg3, %15] : memref<6x6xf64>
      } {"Emitted from" = "linalg.generic"}
      scf.for %arg4 = %14#1 to %13 step %c1 {
        %15 = memref.load %6[%arg4] : memref<?xindex>
        %16 = memref.load %1[%arg4] : memref<?xf64>
        memref.store %16, %2[%arg3, %15] : memref<6x6xf64>
      } {"Emitted from" = "linalg.generic"}
    } {"Emitted from" = "linalg.generic"}
    %7 = bufferization.to_tensor %2 : memref<6x6xf64>
    return %7 : tensor<6x6xf64>
  }
}

