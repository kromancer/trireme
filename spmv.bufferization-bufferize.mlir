// mlir-opt --bufferization-bufferize spmv.func-bufferize.mlir > spmv.bufferization-bufferize.mlir

module {
  func.func private @endLexInsert(!llvm.ptr)
  func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}
  func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
  func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}
  func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

  func.func @spMV(%arg0: !llvm.ptr, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> {
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    %0 = call @sparsePositions0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %1 = call @sparseCoordinates0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %2 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
    
    scf.for %arg3 = %c0 to %c3 step %c1 {
      %3 = memref.load %arg2[%arg3] : memref<3xf64>
      %4 = memref.load %0[%arg3] : memref<?xindex>
      %5 = arith.addi %arg3, %c1 : index
      %6 = memref.load %0[%5] : memref<?xindex>
      %7 = scf.for %arg4 = %4 to %6 step %c1 iter_args(%arg5 = %3) -> (f64) {
        %8 = memref.load %1[%arg4] : memref<?xindex>
        %9 = memref.load %2[%arg4] : memref<?xf64>
        %10 = memref.load %arg1[%8] : memref<4xf64>
        %11 = arith.mulf %9, %10 : f64
        %12 = arith.addf %arg5, %11 : f64
        scf.yield %12 : f64
      }
      
      memref.store %7, %arg2[%arg3] : memref<3xf64>
    }
    
    return %arg2 : memref<3xf64>
  }

  func.func @main(%arg0: memref<3x4xf64>, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
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
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = call @newSparseTensor(%cast_1, %cast_1, %cast, %cast_3, %cast_3, %c0_i32, %c0_i32, %c1_i32, %c0_i32, %0) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    %alloca_4 = memref.alloca() : memref<2xindex>
    %cast_5 = memref.cast %alloca_4 : memref<2xindex> to memref<?xindex>
    %alloca_6 = memref.alloca() : memref<f64>
    scf.for %arg3 = %c0 to %c3 step %c1 {
      scf.for %arg4 = %c0 to %c4 step %c1 {
        %3 = memref.load %arg0[%arg3, %arg4] : memref<3x4xf64>
        %4 = arith.cmpf une, %3, %cst : f64
        scf.if %4 {
          memref.store %arg3, %alloca_4[%c0] : memref<2xindex>
          memref.store %arg4, %alloca_4[%c1] : memref<2xindex>
          memref.store %3, %alloca_6[] : memref<f64>
          func.call @lexInsertF64(%1, %cast_5, %alloca_6) : (!llvm.ptr, memref<?xindex>, memref<f64>) -> ()
        }
      } {"Emitted from" = "sparse_tensor.foreach"}
    } {"Emitted from" = "sparse_tensor.foreach"}
    call @endLexInsert(%1) : (!llvm.ptr) -> ()
    %2 = call @spMV(%1, %arg1, %arg2) : (!llvm.ptr, memref<4xf64>, memref<3xf64>) -> memref<3xf64>
    return %2 : memref<3xf64>
  }
}

