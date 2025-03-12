#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
#sparse1 = #sparse_tensor.encoding<{ map = (d0, d1) -> (d1 : dense, d0 : compressed) }>
module {
  func.func @mat_add(%arg0: tensor<6x6xf64>, %arg1: tensor<6x6xf64, #sparse>, %arg2: tensor<6x6xf64, #sparse1>) -> tensor<6x6xf64> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = sparse_tensor.reinterpret_map %arg2 : tensor<6x6xf64, #sparse1> to tensor<6x6xf64, #sparse>
    %1 = sparse_tensor.convert %arg1 : tensor<6x6xf64, #sparse> to tensor<6x6xf64, #sparse1>
    %2 = sparse_tensor.reinterpret_map %1 : tensor<6x6xf64, #sparse1> to tensor<6x6xf64, #sparse>
    %3 = sparse_tensor.values %2 : tensor<6x6xf64, #sparse> to memref<?xf64>
    %4 = sparse_tensor.values %0 : tensor<6x6xf64, #sparse> to memref<?xf64>
    %5 = bufferization.to_memref %arg0 : memref<6x6xf64>
    linalg.fill ins(%cst : f64) outs(%5 : memref<6x6xf64>)
    %6 = sparse_tensor.positions %2 {level = 1 : index} : tensor<6x6xf64, #sparse> to memref<?xindex>
    %7 = sparse_tensor.coordinates %2 {level = 1 : index} : tensor<6x6xf64, #sparse> to memref<?xindex>
    %8 = sparse_tensor.positions %0 {level = 1 : index} : tensor<6x6xf64, #sparse> to memref<?xindex>
    %9 = sparse_tensor.coordinates %0 {level = 1 : index} : tensor<6x6xf64, #sparse> to memref<?xindex>
    scf.for %arg3 = %c0 to %c6 step %c1 {
      %11 = memref.load %6[%arg3] : memref<?xindex>
      %12 = arith.addi %arg3, %c1 : index
      %13 = memref.load %6[%12] : memref<?xindex>
      %14 = memref.load %8[%arg3] : memref<?xindex>
      %15 = arith.addi %arg3, %c1 : index
      %16 = memref.load %8[%15] : memref<?xindex>
      %17:2 = scf.while (%arg4 = %11, %arg5 = %14) : (index, index) -> (index, index) {
        %18 = arith.cmpi ult, %arg4, %13 : index
        %19 = arith.cmpi ult, %arg5, %16 : index
        %20 = arith.andi %18, %19 : i1
        scf.condition(%20) %arg4, %arg5 : index, index
      } do {
      ^bb0(%arg4: index, %arg5: index):
        %18 = memref.load %7[%arg4] : memref<?xindex>
        %19 = memref.load %9[%arg5] : memref<?xindex>
        %20 = arith.cmpi ult, %19, %18 : index
        %21 = arith.select %20, %19, %18 : index
        %22 = arith.cmpi eq, %18, %21 : index
        %23 = arith.cmpi eq, %19, %21 : index
        %24 = arith.andi %22, %23 : i1
        scf.if %24 {
          %31 = memref.load %3[%arg4] : memref<?xf64>
          %32 = memref.load %4[%arg5] : memref<?xf64>
          %33 = arith.addf %31, %32 : f64
          memref.store %33, %5[%21, %arg3] : memref<6x6xf64>
        } else {
          %31 = arith.cmpi eq, %18, %21 : index
          scf.if %31 {
            %32 = memref.load %3[%arg4] : memref<?xf64>
            memref.store %32, %5[%21, %arg3] : memref<6x6xf64>
          } else {
            %32 = arith.cmpi eq, %19, %21 : index
            scf.if %32 {
              %33 = memref.load %4[%arg5] : memref<?xf64>
              memref.store %33, %5[%21, %arg3] : memref<6x6xf64>
            } else {
            }
          }
        }
        %25 = arith.cmpi eq, %18, %21 : index
        %26 = arith.addi %arg4, %c1 : index
        %27 = arith.select %25, %26, %arg4 : index
        %28 = arith.cmpi eq, %19, %21 : index
        %29 = arith.addi %arg5, %c1 : index
        %30 = arith.select %28, %29, %arg5 : index
        scf.yield %27, %30 : index, index
      } attributes {"Emitted from" = "linalg.generic"}
      scf.for %arg4 = %17#0 to %13 step %c1 {
        %18 = memref.load %7[%arg4] : memref<?xindex>
        %19 = memref.load %3[%arg4] : memref<?xf64>
        memref.store %19, %5[%18, %arg3] : memref<6x6xf64>
      } {"Emitted from" = "linalg.generic"}
      scf.for %arg4 = %17#1 to %16 step %c1 {
        %18 = memref.load %9[%arg4] : memref<?xindex>
        %19 = memref.load %4[%arg4] : memref<?xf64>
        memref.store %19, %5[%18, %arg3] : memref<6x6xf64>
      } {"Emitted from" = "linalg.generic"}
    } {"Emitted from" = "linalg.generic"}
    %10 = bufferization.to_tensor %5 : memref<6x6xf64>
    bufferization.dealloc_tensor %1 : tensor<6x6xf64, #sparse1>
    return %10 : tensor<6x6xf64>
  }
}

