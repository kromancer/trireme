// mlir-opt --sparse-tensor-conversion --canonicalize --tensor-bufferize spmv.sparsification.mlir > spmv.tensor-bufferize.mlir

module {
  func.func private @endLexInsert(!llvm.ptr)
  func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}
  func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
  func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}
  func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func @spMV(%arg0: !llvm.ptr, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %0 = bufferization.to_memref %arg1 : memref<4xf64>
    %1 = bufferization.to_memref %arg2 : memref<3xf64>
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %2 = call @sparsePositions0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %3 = call @sparseCoordinates0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %4 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
    %5 = bufferization.to_memref %arg2 : memref<3xf64>
    scf.for %arg3 = %c0 to %c3 step %c1 {
      %7 = memref.load %1[%arg3] : memref<3xf64>
      %8 = memref.load %2[%arg3] : memref<?xindex>
      %9 = arith.addi %arg3, %c1 : index
      %10 = memref.load %2[%9] : memref<?xindex>
      %11 = scf.for %arg4 = %8 to %10 step %c1 iter_args(%arg5 = %7) -> (f64) {
        %12 = memref.load %3[%arg4] : memref<?xindex>
        %13 = memref.load %4[%arg4] : memref<?xf64>
        %14 = memref.load %0[%12] : memref<4xf64>
        %15 = arith.mulf %13, %14 : f64
        %16 = arith.addf %arg5, %15 : f64
        scf.yield %16 : f64
      }
      memref.store %11, %5[%arg3] : memref<3xf64>
    }
    %6 = bufferization.to_tensor %5 : memref<3xf64>
    return %6 : tensor<3xf64>
  }
  func.func @main(%arg0: tensor<3x4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_memref %arg0 : memref<3x4xf64>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %c4_i8 = arith.constant 4 : i8
    %c8_i8 = arith.constant 8 : i8
    %alloca = memref.alloca() : memref<2xi8>
    %cast = memref.cast %alloca : memref<2xi8> to memref<?xi8>
    memref.store %c4_i8, %alloca[%c0] : memref<2xi8>
    memref.store %c8_i8, %alloca[%c1] : memref<2xi8>
    %alloca_0 = memref.alloca() : memref<2xindex>
    %cast_1 = memref.cast %alloca_0 : memref<2xindex> to memref<?xindex>
    memref.store %c3, %alloca_0[%c0] : memref<2xindex>
    memref.store %c4, %alloca_0[%c1] : memref<2xindex>
    %alloca_2 = memref.alloca() : memref<2xindex>
    %cast_3 = memref.cast %alloca_2 : memref<2xindex> to memref<?xindex>
    memref.store %c0, %alloca_2[%c0] : memref<2xindex>
    memref.store %c1, %alloca_2[%c1] : memref<2xindex>
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = call @newSparseTensor(%cast_1, %cast_1, %cast, %cast_3, %cast_3, %c0_i32, %c0_i32, %c1_i32, %c0_i32, %1) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    %alloca_4 = memref.alloca() : memref<2xindex>
    %cast_5 = memref.cast %alloca_4 : memref<2xindex> to memref<?xindex>
    %alloca_6 = memref.alloca() : memref<f64>
    scf.for %arg3 = %c0 to %c3 step %c1 {
      scf.for %arg4 = %c0 to %c4 step %c1 {
        %4 = memref.load %0[%arg3, %arg4] : memref<3x4xf64>
        %5 = arith.cmpf une, %4, %cst : f64
        scf.if %5 {
          memref.store %arg3, %alloca_4[%c0] : memref<2xindex>
          memref.store %arg4, %alloca_4[%c1] : memref<2xindex>
          memref.store %4, %alloca_6[] : memref<f64>
          func.call @lexInsertF64(%2, %cast_5, %alloca_6) : (!llvm.ptr, memref<?xindex>, memref<f64>) -> ()
        }
      } {"Emitted from" = "sparse_tensor.foreach"}
    } {"Emitted from" = "sparse_tensor.foreach"}
    call @endLexInsert(%2) : (!llvm.ptr) -> ()
    %3 = call @spMV(%2, %arg1, %arg2) : (!llvm.ptr, tensor<4xf64>, tensor<3xf64>) -> tensor<3xf64>
    return %3 : tensor<3xf64>
  }
}

