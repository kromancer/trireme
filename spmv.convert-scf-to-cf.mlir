// mlir-opt --convert-scf-to-cf spmv.bufferization-bufferize.mlir > spmv.convert-scf-to-cf.mlir

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
    cf.br ^bb1(%c0 : index)
  ^bb1(%3: index):  // 2 preds: ^bb0, ^bb5
    %4 = arith.cmpi slt, %3, %c3 : index
    cf.cond_br %4, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %5 = memref.load %arg2[%3] : memref<3xf64>
    %6 = memref.load %0[%3] : memref<?xindex>
    %7 = arith.addi %3, %c1 : index
    %8 = memref.load %0[%7] : memref<?xindex>
    cf.br ^bb3(%6, %5 : index, f64)
  ^bb3(%9: index, %10: f64):  // 2 preds: ^bb2, ^bb4
    %11 = arith.cmpi slt, %9, %8 : index
    cf.cond_br %11, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %12 = memref.load %1[%9] : memref<?xindex>
    %13 = memref.load %2[%9] : memref<?xf64>
    %14 = memref.load %arg1[%12] : memref<4xf64>
    %15 = arith.mulf %13, %14 : f64
    %16 = arith.addf %10, %15 : f64
    %17 = arith.addi %9, %c1 : index
    cf.br ^bb3(%17, %16 : index, f64)
  ^bb5:  // pred: ^bb3
    memref.store %10, %arg2[%3] : memref<3xf64>
    %18 = arith.addi %3, %c1 : index
    cf.br ^bb1(%18 : index)
  ^bb6:  // pred: ^bb1
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
    cf.br ^bb1(%c0 : index)
  ^bb1(%2: index):  // 2 preds: ^bb0, ^bb7
    %3 = arith.cmpi slt, %2, %c3 : index
    cf.cond_br %3, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%4: index):  // 2 preds: ^bb2, ^bb6
    %5 = arith.cmpi slt, %4, %c4 : index
    cf.cond_br %5, ^bb4, ^bb7
  ^bb4:  // pred: ^bb3
    %6 = memref.load %arg0[%2, %4] : memref<3x4xf64>
    %7 = arith.cmpf une, %6, %cst : f64
    cf.cond_br %7, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    memref.store %2, %alloca_4[%c0] : memref<2xindex>
    memref.store %4, %alloca_4[%c1] : memref<2xindex>
    memref.store %6, %alloca_6[] : memref<f64>
    call @lexInsertF64(%1, %cast_5, %alloca_6) : (!llvm.ptr, memref<?xindex>, memref<f64>) -> ()
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %8 = arith.addi %4, %c1 : index
    cf.br ^bb3(%8 : index)
  ^bb7:  // pred: ^bb3
    %9 = arith.addi %2, %c1 : index
    cf.br ^bb1(%9 : index)
  ^bb8:  // pred: ^bb1
    call @endLexInsert(%1) : (!llvm.ptr) -> ()
    %10 = call @spMV(%1, %arg1, %arg2) : (!llvm.ptr, memref<4xf64>, memref<3xf64>) -> memref<3xf64>
    return %10 : memref<3xf64>
  }
}

