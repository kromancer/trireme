// mlir-opt --sparse-tensor-conversion spmv.sparsification.mlir > spmv.sparse-tensor-conversion.mlir

module {
  func.func private @endLexInsert(!llvm.ptr)
  func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}
  func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
  func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}
  func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func @spMV(%arg0: !llvm.ptr, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %0 = call @sparsePositions0(%arg0, %c1_0) : (!llvm.ptr, index) -> memref<?xindex>
    %c1_1 = arith.constant 1 : index
    %1 = call @sparseCoordinates0(%arg0, %c1_1) : (!llvm.ptr, index) -> memref<?xindex>
    %2 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
    %3 = bufferization.to_memref %arg1 : memref<4xf64>
    %4 = bufferization.to_memref %arg2 : memref<3xf64>
    scf.for %arg3 = %c0 to %c3 step %c1 {
      %6 = memref.load %4[%arg3] : memref<3xf64>
      %7 = memref.load %0[%arg3] : memref<?xindex>
      %8 = arith.addi %arg3, %c1 : index
      %9 = memref.load %0[%8] : memref<?xindex>
      %10 = scf.for %arg4 = %7 to %9 step %c1 iter_args(%arg5 = %6) -> (f64) {
        %11 = memref.load %1[%arg4] : memref<?xindex>
        %12 = memref.load %2[%arg4] : memref<?xf64>
        %13 = memref.load %3[%11] : memref<4xf64>
        %14 = arith.mulf %12, %13 : f64
        %15 = arith.addf %arg5, %14 : f64
        scf.yield %15 : f64
      }
      memref.store %10, %4[%arg3] : memref<3xf64>
    }
    %5 = bufferization.to_tensor %4 : memref<3xf64>
    return %5 : tensor<3xf64>
  }
  func.func @main(%arg0: tensor<3x4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> attributes {llvm.emit_c_interface} {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %c3_0 = arith.constant 3 : index
    %c4_1 = arith.constant 4 : index
    %c4_i8 = arith.constant 4 : i8
    %c8_i8 = arith.constant 8 : i8
    %c2 = arith.constant 2 : index
    %alloca = memref.alloca(%c2) : memref<?xi8>
    %c0_2 = arith.constant 0 : index
    memref.store %c4_i8, %alloca[%c0_2] : memref<?xi8>
    %c1_3 = arith.constant 1 : index
    memref.store %c8_i8, %alloca[%c1_3] : memref<?xi8>
    %c2_4 = arith.constant 2 : index
    %alloca_5 = memref.alloca(%c2_4) : memref<?xindex>
    %c0_6 = arith.constant 0 : index
    memref.store %c3_0, %alloca_5[%c0_6] : memref<?xindex>
    %c1_7 = arith.constant 1 : index
    memref.store %c4_1, %alloca_5[%c1_7] : memref<?xindex>
    %c0_8 = arith.constant 0 : index
    %c1_9 = arith.constant 1 : index
    %c2_10 = arith.constant 2 : index
    %alloca_11 = memref.alloca(%c2_10) : memref<?xindex>
    %c0_12 = arith.constant 0 : index
    memref.store %c0_8, %alloca_11[%c0_12] : memref<?xindex>
    %c1_13 = arith.constant 1 : index
    memref.store %c1_9, %alloca_11[%c1_13] : memref<?xindex>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_14 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32_15 = arith.constant 0 : i32
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = call @newSparseTensor(%alloca_5, %alloca_5, %alloca, %alloca_11, %alloca_11, %c0_i32, %c0_i32_14, %c1_i32, %c0_i32_15, %0) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    %2 = bufferization.to_memref %arg0 : memref<3x4xf64>
    %c2_16 = arith.constant 2 : index
    %alloca_17 = memref.alloca(%c2_16) : memref<?xindex>
    %alloca_18 = memref.alloca() : memref<f64>
    %3 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %1) -> (!llvm.ptr) {
      %5 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %arg4) -> (!llvm.ptr) {
        %6 = memref.load %2[%arg3, %arg5] : memref<3x4xf64>
        %7 = arith.cmpf une, %6, %cst : f64
        %8 = scf.if %7 -> (!llvm.ptr) {
          %c0_19 = arith.constant 0 : index
          memref.store %arg3, %alloca_17[%c0_19] : memref<?xindex>
          %c1_20 = arith.constant 1 : index
          memref.store %arg5, %alloca_17[%c1_20] : memref<?xindex>
          memref.store %6, %alloca_18[] : memref<f64>
          func.call @lexInsertF64(%arg6, %alloca_17, %alloca_18) : (!llvm.ptr, memref<?xindex>, memref<f64>) -> ()
          scf.yield %arg6 : !llvm.ptr
        } else {
          scf.yield %arg6 : !llvm.ptr
        }
        scf.yield %8 : !llvm.ptr
      } {"Emitted from" = "sparse_tensor.foreach"}
      scf.yield %5 : !llvm.ptr
    } {"Emitted from" = "sparse_tensor.foreach"}
    call @endLexInsert(%3) : (!llvm.ptr) -> ()
    %4 = call @spMV(%3, %arg1, %arg2) : (!llvm.ptr, tensor<4xf64>, tensor<3xf64>) -> tensor<3xf64>
    return %4 : tensor<3xf64>
  }
}

