module {
  llvm.func @endLexInsert(!llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func private @lexInsertF64(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: i64) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %5, %7 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %9 = llvm.insertvalue %arg6, %8[0] : !llvm.struct<(ptr, ptr, i64)> 
    %10 = llvm.insertvalue %arg7, %9[1] : !llvm.struct<(ptr, ptr, i64)> 
    %11 = llvm.insertvalue %arg8, %10[2] : !llvm.struct<(ptr, ptr, i64)> 
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.alloca %12 x !llvm.struct<(ptr, ptr, i64)> : (i64) -> !llvm.ptr
    llvm.store %11, %13 : !llvm.struct<(ptr, ptr, i64)>, !llvm.ptr
    llvm.call @_mlir_ciface_lexInsertF64(%arg0, %7, %13) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_lexInsertF64(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @newSparseTensor(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr, %arg16: !llvm.ptr, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %5, %7 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg5, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg6, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg7, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg8, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg9, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.alloca %14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %13, %15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %16 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.insertvalue %arg10, %16[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.insertvalue %arg11, %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.insertvalue %arg12, %18[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.insertvalue %arg13, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.insertvalue %arg14, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.mlir.constant(1 : index) : i64
    %23 = llvm.alloca %22 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %21, %23 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %24 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.insertvalue %arg15, %24[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.insertvalue %arg16, %25[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.insertvalue %arg17, %26[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.insertvalue %arg18, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.insertvalue %arg19, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.alloca %30 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %29, %31 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %32 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %33 = llvm.insertvalue %arg20, %32[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.insertvalue %arg21, %33[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.insertvalue %arg22, %34[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.insertvalue %arg23, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.insertvalue %arg24, %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %38 = llvm.mlir.constant(1 : index) : i64
    %39 = llvm.alloca %38 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %37, %39 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %40 = llvm.call @_mlir_ciface_newSparseTensor(%7, %15, %23, %31, %39, %arg25, %arg26, %arg27, %arg28, %arg29) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    llvm.return %40 : !llvm.ptr
  }
  llvm.func @_mlir_ciface_newSparseTensor(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @sparseValuesF64(%arg0: !llvm.ptr) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.call @_mlir_ciface_sparseValuesF64(%1, %arg0) : (!llvm.ptr, !llvm.ptr) -> ()
    %2 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %2 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @_mlir_ciface_sparseValuesF64(!llvm.ptr, !llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @sparseCoordinates0(%arg0: !llvm.ptr, %arg1: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.call @_mlir_ciface_sparseCoordinates0(%1, %arg0, %arg1) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %2 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %2 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @_mlir_ciface_sparseCoordinates0(!llvm.ptr, !llvm.ptr, i64) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @sparsePositions0(%arg0: !llvm.ptr, %arg1: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.call @_mlir_ciface_sparsePositions0(%1, %arg0, %arg1) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %2 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %2 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @_mlir_ciface_sparsePositions0(!llvm.ptr, !llvm.ptr, i64) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @spMV(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: i64, %arg9: i64, %arg10: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.insertvalue %arg7, %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %arg8, %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg9, %9[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg10, %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.mlir.constant(3 : index) : i64
    %15 = llvm.call @sparsePositions0(%arg0, %12) : (!llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.call @sparseCoordinates0(%arg0, %12) : (!llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.call @sparseValuesF64(%arg0) : (!llvm.ptr) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.br ^bb1(%13 : i64)
  ^bb1(%18: i64):  // 2 preds: ^bb0, ^bb5
    %19 = llvm.icmp "slt" %18, %14 : i64
    llvm.cond_br %19, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %20 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.getelementptr %20[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %22 = llvm.load %21 : !llvm.ptr -> f64
    %23 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.getelementptr %23[%18] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %25 = llvm.load %24 : !llvm.ptr -> i64
    %26 = llvm.add %18, %12  : i64
    %27 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.getelementptr %27[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %29 = llvm.load %28 : !llvm.ptr -> i64
    llvm.br ^bb3(%25, %22 : i64, f64)
  ^bb3(%30: i64, %31: f64):  // 2 preds: ^bb2, ^bb4
    %32 = llvm.icmp "slt" %30, %29 : i64
    llvm.cond_br %32, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %33 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.getelementptr %33[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %35 = llvm.load %34 : !llvm.ptr -> i64
    %36 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.getelementptr %36[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %38 = llvm.load %37 : !llvm.ptr -> f64
    %39 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.getelementptr %39[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %41 = llvm.load %40 : !llvm.ptr -> f64
    %42 = llvm.fmul %38, %41  : f64
    %43 = llvm.fadd %31, %42  : f64
    %44 = llvm.add %30, %12  : i64
    llvm.br ^bb3(%44, %43 : i64, f64)
  ^bb5:  // pred: ^bb3
    %45 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.getelementptr %45[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %31, %46 : f64, !llvm.ptr
    %47 = llvm.add %18, %12  : i64
    llvm.br ^bb1(%47 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return %11 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: i64, %arg16: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg11, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.insertvalue %arg12, %14[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.insertvalue %arg13, %15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.insertvalue %arg14, %16[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.insertvalue %arg15, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.insertvalue %arg16, %18[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.mlir.constant(4 : index) : i64
    %21 = llvm.mlir.constant(0 : index) : i64
    %22 = llvm.mlir.constant(1 : index) : i64
    %23 = llvm.mlir.constant(2 : index) : i64
    %24 = llvm.mlir.constant(8 : i8) : i8
    %25 = llvm.mlir.constant(4 : i8) : i8
    %26 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %27 = llvm.mlir.constant(1 : index) : i64
    %28 = llvm.mlir.constant(0 : index) : i64
    %29 = llvm.mlir.constant(4 : index) : i64
    %30 = llvm.mlir.constant(3 : index) : i64
    %31 = llvm.mlir.constant(0 : i32) : i32
    %32 = llvm.mlir.constant(1 : i32) : i32
    %33 = llvm.alloca %23 x i8 : (i64) -> !llvm.ptr
    %34 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %35 = llvm.insertvalue %33, %34[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.insertvalue %33, %35[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.insertvalue %21, %36[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %38 = llvm.insertvalue %23, %37[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %39 = llvm.insertvalue %22, %38[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.getelementptr %33[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %25, %40 : i8, !llvm.ptr
    %41 = llvm.getelementptr %33[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %24, %41 : i8, !llvm.ptr
    %42 = llvm.alloca %23 x i64 : (i64) -> !llvm.ptr
    %43 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %44 = llvm.insertvalue %42, %43[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.insertvalue %42, %44[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.insertvalue %21, %45[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.insertvalue %23, %46[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %48 = llvm.insertvalue %22, %47[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %49 = llvm.getelementptr %42[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %30, %49 : i64, !llvm.ptr
    %50 = llvm.getelementptr %42[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %29, %50 : i64, !llvm.ptr
    %51 = llvm.alloca %23 x i64 : (i64) -> !llvm.ptr
    %52 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %53 = llvm.insertvalue %51, %52[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %54 = llvm.insertvalue %51, %53[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.insertvalue %21, %54[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = llvm.insertvalue %23, %55[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = llvm.insertvalue %22, %56[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.getelementptr %51[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %28, %58 : i64, !llvm.ptr
    %59 = llvm.getelementptr %51[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %27, %59 : i64, !llvm.ptr
    %60 = llvm.mlir.zero : !llvm.ptr
    %61 = llvm.call @newSparseTensor(%42, %42, %21, %23, %22, %42, %42, %21, %23, %22, %33, %33, %21, %23, %22, %51, %51, %21, %23, %22, %51, %51, %21, %23, %22, %31, %31, %32, %31, %60) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    %62 = llvm.alloca %23 x i64 : (i64) -> !llvm.ptr
    %63 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %64 = llvm.insertvalue %62, %63[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %65 = llvm.insertvalue %62, %64[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %66 = llvm.insertvalue %21, %65[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %67 = llvm.insertvalue %23, %66[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %68 = llvm.insertvalue %22, %67[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %69 = llvm.alloca %22 x f64 : (i64) -> !llvm.ptr
    %70 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %71 = llvm.insertvalue %69, %70[0] : !llvm.struct<(ptr, ptr, i64)> 
    %72 = llvm.insertvalue %69, %71[1] : !llvm.struct<(ptr, ptr, i64)> 
    %73 = llvm.insertvalue %21, %72[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.br ^bb1(%28 : i64)
  ^bb1(%74: i64):  // 2 preds: ^bb0, ^bb7
    %75 = llvm.icmp "slt" %74, %30 : i64
    llvm.cond_br %75, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%28 : i64)
  ^bb3(%76: i64):  // 2 preds: ^bb2, ^bb6
    %77 = llvm.icmp "slt" %76, %29 : i64
    llvm.cond_br %77, ^bb4, ^bb7
  ^bb4:  // pred: ^bb3
    %78 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.mul %74, %20  : i64
    %80 = llvm.add %79, %76  : i64
    %81 = llvm.getelementptr %78[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %82 = llvm.load %81 : !llvm.ptr -> f64
    %83 = llvm.fcmp "une" %82, %26 : f64
    llvm.cond_br %83, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %84 = llvm.getelementptr %62[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %74, %84 : i64, !llvm.ptr
    %85 = llvm.getelementptr %62[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %76, %85 : i64, !llvm.ptr
    llvm.store %82, %69 : f64, !llvm.ptr
    llvm.call @lexInsertF64(%61, %62, %62, %21, %23, %22, %69, %69, %21) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %86 = llvm.add %76, %27  : i64
    llvm.br ^bb3(%86 : i64)
  ^bb7:  // pred: ^bb3
    %87 = llvm.add %74, %27  : i64
    llvm.br ^bb1(%87 : i64)
  ^bb8:  // pred: ^bb1
    llvm.call @endLexInsert(%61) : (!llvm.ptr) -> ()
    %88 = llvm.call @spMV(%61, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %88 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @_mlir_ciface_main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.load %arg3 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.extractvalue %14[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.extractvalue %14[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.extractvalue %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.extractvalue %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.call @main(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %15, %16, %17, %18, %19) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.store %20, %arg0 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.return
  }
}

// -----// IR Dump After LinalgGeneralization (linalg-generalize-named-ops) //----- //
func.func @main(%arg0: tensor<3x4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> attributes {llvm.emit_c_interface} {
  %0 = sparse_tensor.convert %arg0 : tensor<3x4xf64> to tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
  %1 = call @spMV(%0, %arg1, %arg2) : (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, tensor<4xf64>, tensor<3xf64>) -> tensor<3xf64>
  return %1 : tensor<3xf64>
}

// -----// IR Dump After LinalgGeneralization (linalg-generalize-named-ops) //----- //
func.func @spMV(%arg0: tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>], iterator_types = ["reduction", "parallel"]} ins(%arg0, %arg1 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, tensor<4xf64>) outs(%arg2 : tensor<3xf64>) {
  ^bb0(%in: f64, %in_0: f64, %out: f64):
    %1 = arith.mulf %in, %in_0 : f64
    %2 = arith.addf %out, %1 : f64
    linalg.yield %2 : f64
  } -> tensor<3xf64>
  return %0 : tensor<3xf64>
}

// -----// IR Dump After PreSparsificationRewrite (pre-sparsification-rewrite) //----- //
#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @spMV(%arg0: tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel"]} ins(%arg0, %arg1 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, tensor<4xf64>) outs(%arg2 : tensor<3xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
  func.func @main(%arg0: tensor<3x4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> attributes {llvm.emit_c_interface} {
    %0 = sparse_tensor.convert %arg0 : tensor<3x4xf64> to tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
    %1 = call @spMV(%0, %arg1, %arg2) : (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, tensor<4xf64>, tensor<3xf64>) -> tensor<3xf64>
    return %1 : tensor<3xf64>
  }
}


// -----// IR Dump After EmptyTensorToAllocTensor (empty-tensor-to-alloc-tensor) //----- //
func.func @main(%arg0: tensor<3x4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> attributes {llvm.emit_c_interface} {
  %0 = sparse_tensor.convert %arg0 : tensor<3x4xf64> to tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
  %1 = call @spMV(%0, %arg1, %arg2) : (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, tensor<4xf64>, tensor<3xf64>) -> tensor<3xf64>
  return %1 : tensor<3xf64>
}

// -----// IR Dump After EmptyTensorToAllocTensor (empty-tensor-to-alloc-tensor) //----- //
func.func @spMV(%arg0: tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>], iterator_types = ["reduction", "parallel"]} ins(%arg0, %arg1 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, tensor<4xf64>) outs(%arg2 : tensor<3xf64>) {
  ^bb0(%in: f64, %in_0: f64, %out: f64):
    %1 = arith.mulf %in, %in_0 : f64
    %2 = arith.addf %out, %1 : f64
    linalg.yield %2 : f64
  } -> tensor<3xf64>
  return %0 : tensor<3xf64>
}

// -----// IR Dump After SparseReinterpretMap (sparse-reinterpret-map) //----- //
#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @spMV(%arg0: tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel"]} ins(%arg0, %arg1 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, tensor<4xf64>) outs(%arg2 : tensor<3xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
  func.func @main(%arg0: tensor<3x4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> attributes {llvm.emit_c_interface} {
    %0 = sparse_tensor.convert %arg0 : tensor<3x4xf64> to tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
    %1 = call @spMV(%0, %arg1, %arg2) : (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, tensor<4xf64>, tensor<3xf64>) -> tensor<3xf64>
    return %1 : tensor<3xf64>
  }
}


// -----// IR Dump After SparsificationPass (sparsification) //----- //
module {
  func.func @spMV(%arg0: tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = sparse_tensor.positions %arg0 {level = 1 : index} : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xindex>
    %1 = sparse_tensor.coordinates %arg0 {level = 1 : index} : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xindex>
    %2 = sparse_tensor.values %arg0 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xf64>
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
      } {"Emitted from" = "linalg.generic"}
      memref.store %10, %4[%arg3] : memref<3xf64>
    } {"Emitted from" = "linalg.generic"}
    %5 = bufferization.to_tensor %4 : memref<3xf64>
    return %5 : tensor<3xf64>
  }
  func.func @main(%arg0: tensor<3x4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> attributes {llvm.emit_c_interface} {
    %0 = sparse_tensor.convert %arg0 : tensor<3x4xf64> to tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
    %1 = call @spMV(%0, %arg1, %arg2) : (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, tensor<4xf64>, tensor<3xf64>) -> tensor<3xf64>
    return %1 : tensor<3xf64>
  }
}


// -----// IR Dump After StageSparseOperations (stage-sparse-ops) //----- //
func.func @main(%arg0: tensor<3x4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> attributes {llvm.emit_c_interface} {
  %0 = sparse_tensor.convert %arg0 : tensor<3x4xf64> to tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
  %1 = call @spMV(%0, %arg1, %arg2) : (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, tensor<4xf64>, tensor<3xf64>) -> tensor<3xf64>
  return %1 : tensor<3xf64>
}

// -----// IR Dump After StageSparseOperations (stage-sparse-ops) //----- //
func.func @spMV(%arg0: tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = sparse_tensor.positions %arg0 {level = 1 : index} : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xindex>
  %1 = sparse_tensor.coordinates %arg0 {level = 1 : index} : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xindex>
  %2 = sparse_tensor.values %arg0 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xf64>
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
    } {"Emitted from" = "linalg.generic"}
    memref.store %10, %4[%arg3] : memref<3xf64>
  } {"Emitted from" = "linalg.generic"}
  %5 = bufferization.to_tensor %4 : memref<3xf64>
  return %5 : tensor<3xf64>
}

// -----// IR Dump After LowerSparseOpsToForeach (lower-sparse-ops-to-foreach) //----- //
module {
  func.func @spMV(%arg0: tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = sparse_tensor.positions %arg0 {level = 1 : index} : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xindex>
    %1 = sparse_tensor.coordinates %arg0 {level = 1 : index} : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xindex>
    %2 = sparse_tensor.values %arg0 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xf64>
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
      } {"Emitted from" = "linalg.generic"}
      memref.store %10, %4[%arg3] : memref<3xf64>
    } {"Emitted from" = "linalg.generic"}
    %5 = bufferization.to_tensor %4 : memref<3xf64>
    return %5 : tensor<3xf64>
  }
  func.func @main(%arg0: tensor<3x4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.alloc_tensor() : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
    %1 = sparse_tensor.foreach in %arg0 init(%0) : tensor<3x4xf64>, tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> -> tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> do {
    ^bb0(%arg3: index, %arg4: index, %arg5: f64, %arg6: tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>):
      %4 = arith.cmpf une, %arg5, %cst : f64
      %5 = scf.if %4 -> (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>) {
        %inserted = tensor.insert %arg5 into %arg6[%arg3, %arg4] : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
        scf.yield %inserted : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
      } else {
        scf.yield %arg6 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
      }
      sparse_tensor.yield %5 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
    }
    %2 = sparse_tensor.load %1 hasInserts : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
    %3 = call @spMV(%2, %arg1, %arg2) : (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, tensor<4xf64>, tensor<3xf64>) -> tensor<3xf64>
    return %3 : tensor<3xf64>
  }
}


// -----// IR Dump After SparseReinterpretMap (sparse-reinterpret-map) //----- //
module {
  func.func @spMV(%arg0: tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = sparse_tensor.positions %arg0 {level = 1 : index} : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xindex>
    %1 = sparse_tensor.coordinates %arg0 {level = 1 : index} : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xindex>
    %2 = sparse_tensor.values %arg0 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xf64>
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
      } {"Emitted from" = "linalg.generic"}
      memref.store %10, %4[%arg3] : memref<3xf64>
    } {"Emitted from" = "linalg.generic"}
    %5 = bufferization.to_tensor %4 : memref<3xf64>
    return %5 : tensor<3xf64>
  }
  func.func @main(%arg0: tensor<3x4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.alloc_tensor() : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
    %1 = sparse_tensor.foreach in %arg0 init(%0) : tensor<3x4xf64>, tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> -> tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> do {
    ^bb0(%arg3: index, %arg4: index, %arg5: f64, %arg6: tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>):
      %4 = arith.cmpf une, %arg5, %cst : f64
      %5 = scf.if %4 -> (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>) {
        %6 = sparse_tensor.insert %arg5 into %arg6[%arg3, %arg4] : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
        scf.yield %6 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
      } else {
        scf.yield %arg6 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
      }
      sparse_tensor.yield %5 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
    }
    %2 = sparse_tensor.load %1 hasInserts : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
    %3 = call @spMV(%2, %arg1, %arg2) : (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, tensor<4xf64>, tensor<3xf64>) -> tensor<3xf64>
    return %3 : tensor<3xf64>
  }
}


// -----// IR Dump After LowerForeachToSCF (lower-sparse-foreach-to-scf) //----- //
func.func @spMV(%arg0: tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = sparse_tensor.positions %arg0 {level = 1 : index} : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xindex>
  %1 = sparse_tensor.coordinates %arg0 {level = 1 : index} : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xindex>
  %2 = sparse_tensor.values %arg0 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xf64>
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
    } {"Emitted from" = "linalg.generic"}
    memref.store %10, %4[%arg3] : memref<3xf64>
  } {"Emitted from" = "linalg.generic"}
  %5 = bufferization.to_tensor %4 : memref<3xf64>
  return %5 : tensor<3xf64>
}

// -----// IR Dump After LowerForeachToSCF (lower-sparse-foreach-to-scf) //----- //
func.func @main(%arg0: tensor<3x4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> attributes {llvm.emit_c_interface} {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %0 = bufferization.alloc_tensor() : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
  %1 = bufferization.to_memref %arg0 : memref<3x4xf64>
  %2 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %0) -> (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>) {
    %5 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %arg4) -> (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>) {
      %6 = memref.load %1[%arg3, %arg5] : memref<3x4xf64>
      %7 = arith.cmpf une, %6, %cst : f64
      %8 = scf.if %7 -> (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>) {
        %9 = sparse_tensor.insert %6 into %arg6[%arg3, %arg5] : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
        scf.yield %9 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
      } else {
        scf.yield %arg6 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
      }
      scf.yield %8 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
    } {"Emitted from" = "sparse_tensor.foreach"}
    scf.yield %5 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
  } {"Emitted from" = "sparse_tensor.foreach"}
  %3 = sparse_tensor.load %2 hasInserts : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
  %4 = call @spMV(%3, %arg1, %arg2) : (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, tensor<4xf64>, tensor<3xf64>) -> tensor<3xf64>
  return %4 : tensor<3xf64>
}

// -----// IR Dump After SparseTensorConversionPass (sparse-tensor-conversion) //----- //
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
      } {"Emitted from" = "linalg.generic"}
      memref.store %10, %4[%arg3] : memref<3xf64>
    } {"Emitted from" = "linalg.generic"}
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


// -----// IR Dump After SparsificationAndBufferization (sparsification-and-bufferization) //----- //
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
    %c1_0 = arith.constant 1 : index
    %0 = call @sparsePositions0(%arg0, %c1_0) : (!llvm.ptr, index) -> memref<?xindex>
    %c1_1 = arith.constant 1 : index
    %1 = call @sparseCoordinates0(%arg0, %c1_1) : (!llvm.ptr, index) -> memref<?xindex>
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
      } {"Emitted from" = "linalg.generic"}
      memref.store %7, %arg2[%arg3] : memref<3xf64>
    } {"Emitted from" = "linalg.generic"}
    return %arg2 : memref<3xf64>
  }
  func.func @main(%arg0: memref<3x4xf64>, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
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
    %c2_16 = arith.constant 2 : index
    %alloca_17 = memref.alloca(%c2_16) : memref<?xindex>
    %alloca_18 = memref.alloca() : memref<f64>
    %2 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %1) -> (!llvm.ptr) {
      %4 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %arg4) -> (!llvm.ptr) {
        %5 = memref.load %arg0[%arg3, %arg5] : memref<3x4xf64>
        %6 = arith.cmpf une, %5, %cst : f64
        %7 = scf.if %6 -> (!llvm.ptr) {
          %c0_19 = arith.constant 0 : index
          memref.store %arg3, %alloca_17[%c0_19] : memref<?xindex>
          %c1_20 = arith.constant 1 : index
          memref.store %arg5, %alloca_17[%c1_20] : memref<?xindex>
          memref.store %5, %alloca_18[] : memref<f64>
          func.call @lexInsertF64(%arg6, %alloca_17, %alloca_18) : (!llvm.ptr, memref<?xindex>, memref<f64>) -> ()
          scf.yield %arg6 : !llvm.ptr
        } else {
          scf.yield %arg6 : !llvm.ptr
        }
        scf.yield %7 : !llvm.ptr
      } {"Emitted from" = "sparse_tensor.foreach"}
      scf.yield %4 : !llvm.ptr
    } {"Emitted from" = "sparse_tensor.foreach"}
    call @endLexInsert(%2) : (!llvm.ptr) -> ()
    %3 = call @spMV(%2, %arg1, %arg2) : (!llvm.ptr, memref<4xf64>, memref<3xf64>) -> memref<3xf64>
    return %3 : memref<3xf64>
  }
}


// -----// IR Dump After StorageSpecifierToLLVM (sparse-storage-specifier-to-llvm) //----- //
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
    %c1_0 = arith.constant 1 : index
    %0 = call @sparsePositions0(%arg0, %c1_0) : (!llvm.ptr, index) -> memref<?xindex>
    %c1_1 = arith.constant 1 : index
    %1 = call @sparseCoordinates0(%arg0, %c1_1) : (!llvm.ptr, index) -> memref<?xindex>
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
      } {"Emitted from" = "linalg.generic"}
      memref.store %7, %arg2[%arg3] : memref<3xf64>
    } {"Emitted from" = "linalg.generic"}
    return %arg2 : memref<3xf64>
  }
  func.func @main(%arg0: memref<3x4xf64>, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
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
    %c2_16 = arith.constant 2 : index
    %alloca_17 = memref.alloca(%c2_16) : memref<?xindex>
    %alloca_18 = memref.alloca() : memref<f64>
    %2 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %1) -> (!llvm.ptr) {
      %4 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %arg4) -> (!llvm.ptr) {
        %5 = memref.load %arg0[%arg3, %arg5] : memref<3x4xf64>
        %6 = arith.cmpf une, %5, %cst : f64
        %7 = scf.if %6 -> (!llvm.ptr) {
          %c0_19 = arith.constant 0 : index
          memref.store %arg3, %alloca_17[%c0_19] : memref<?xindex>
          %c1_20 = arith.constant 1 : index
          memref.store %arg5, %alloca_17[%c1_20] : memref<?xindex>
          memref.store %5, %alloca_18[] : memref<f64>
          func.call @lexInsertF64(%arg6, %alloca_17, %alloca_18) : (!llvm.ptr, memref<?xindex>, memref<f64>) -> ()
          scf.yield %arg6 : !llvm.ptr
        } else {
          scf.yield %arg6 : !llvm.ptr
        }
        scf.yield %7 : !llvm.ptr
      } {"Emitted from" = "sparse_tensor.foreach"}
      scf.yield %4 : !llvm.ptr
    } {"Emitted from" = "sparse_tensor.foreach"}
    call @endLexInsert(%2) : (!llvm.ptr) -> ()
    %3 = call @spMV(%2, %arg1, %arg2) : (!llvm.ptr, memref<4xf64>, memref<3xf64>) -> memref<3xf64>
    return %3 : memref<3xf64>
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func private @endLexInsert(!llvm.ptr)

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After FinalizingBufferize (finalizing-bufferize) //----- //
func.func private @endLexInsert(!llvm.ptr)

// -----// IR Dump After FinalizingBufferize (finalizing-bufferize) //----- //
func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}

// -----// IR Dump After FinalizingBufferize (finalizing-bufferize) //----- //
func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
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
    } {"Emitted from" = "linalg.generic"}
    memref.store %7, %arg2[%arg3] : memref<3xf64>
  } {"Emitted from" = "linalg.generic"}
  return %arg2 : memref<3xf64>
}

// -----// IR Dump After LinalgLowerToLoops (convert-linalg-to-loops) //----- //
func.func private @endLexInsert(!llvm.ptr)

// -----// IR Dump After LinalgLowerToLoops (convert-linalg-to-loops) //----- //
func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}

// -----// IR Dump After LinalgLowerToLoops (convert-linalg-to-loops) //----- //
func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}

// -----// IR Dump After FinalizingBufferize (finalizing-bufferize) //----- //
func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After FinalizingBufferize (finalizing-bufferize) //----- //
func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}

// -----// IR Dump After FinalizingBufferize (finalizing-bufferize) //----- //
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
    } {"Emitted from" = "linalg.generic"}
    memref.store %7, %arg2[%arg3] : memref<3xf64>
  } {"Emitted from" = "linalg.generic"}
  return %arg2 : memref<3xf64>
}

// -----// IR Dump After ConvertVectorToSCF (convert-vector-to-scf) //----- //
func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertVectorToSCF (convert-vector-to-scf) //----- //
func.func private @endLexInsert(!llvm.ptr)

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
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

// -----// IR Dump After ConvertVectorToSCF (convert-vector-to-scf) //----- //
func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}

// -----// IR Dump After LinalgLowerToLoops (convert-linalg-to-loops) //----- //
func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After LinalgLowerToLoops (convert-linalg-to-loops) //----- //
func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}

// -----// IR Dump After FinalizingBufferize (finalizing-bufferize) //----- //
func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}

// -----// IR Dump After LinalgLowerToLoops (convert-linalg-to-loops) //----- //
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
    } {"Emitted from" = "linalg.generic"}
    memref.store %7, %arg2[%arg3] : memref<3xf64>
  } {"Emitted from" = "linalg.generic"}
  return %arg2 : memref<3xf64>
}

// -----// IR Dump After ExpandRealloc (expand-realloc) //----- //
func.func private @endLexInsert(!llvm.ptr)

// -----// IR Dump After LinalgLowerToLoops (convert-linalg-to-loops) //----- //
func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertVectorToSCF (convert-vector-to-scf) //----- //
func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertVectorToSCF (convert-vector-to-scf) //----- //
func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After FinalizingBufferize (finalizing-bufferize) //----- //
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

// -----// IR Dump After ExpandRealloc (expand-realloc) //----- //
func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After ExpandRealloc (expand-realloc) //----- //
func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertVectorToSCF (convert-vector-to-scf) //----- //
func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertVectorToSCF (convert-vector-to-scf) //----- //
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
    } {"Emitted from" = "linalg.generic"}
    memref.store %7, %arg2[%arg3] : memref<3xf64>
  } {"Emitted from" = "linalg.generic"}
  return %arg2 : memref<3xf64>
}

// -----// IR Dump After SCFToControlFlow (convert-scf-to-cf) //----- //
func.func private @endLexInsert(!llvm.ptr)

// -----// IR Dump After ExpandRealloc (expand-realloc) //----- //
func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}

// -----// IR Dump After LinalgLowerToLoops (convert-linalg-to-loops) //----- //
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

// -----// IR Dump After ExpandRealloc (expand-realloc) //----- //
func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}

// -----// IR Dump After SCFToControlFlow (convert-scf-to-cf) //----- //
func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}

// -----// IR Dump After ExpandRealloc (expand-realloc) //----- //
func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After ExpandRealloc (expand-realloc) //----- //
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
    } {"Emitted from" = "linalg.generic"}
    memref.store %7, %arg2[%arg3] : memref<3xf64>
  } {"Emitted from" = "linalg.generic"}
  return %arg2 : memref<3xf64>
}

// -----// IR Dump After SCFToControlFlow (convert-scf-to-cf) //----- //
func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertVectorToSCF (convert-vector-to-scf) //----- //
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

// -----// IR Dump After SCFToControlFlow (convert-scf-to-cf) //----- //
func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}

// -----// IR Dump After SCFToControlFlow (convert-scf-to-cf) //----- //
func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}

// -----// IR Dump After SCFToControlFlow (convert-scf-to-cf) //----- //
func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After ExpandRealloc (expand-realloc) //----- //
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

// -----// IR Dump After SCFToControlFlow (convert-scf-to-cf) //----- //
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

// -----// IR Dump After SCFToControlFlow (convert-scf-to-cf) //----- //
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

// -----// IR Dump After ExpandStridedMetadata (expand-strided-metadata) //----- //
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


// -----// IR Dump After ConvertAffineToStandard (lower-affine) //----- //
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


// -----// IR Dump After ConvertVectorToLLVMPass (convert-vector-to-llvm) //----- //
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


// -----// IR Dump After FinalizeMemRefToLLVMConversionPass (finalize-memref-to-llvm) //----- //
module {
  func.func private @endLexInsert(!llvm.ptr)
  func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}
  func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
  func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}
  func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func @spMV(%arg0: !llvm.ptr, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> {
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<3xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<4xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %2 = call @sparsePositions0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %3 = builtin.unrealized_conversion_cast %2 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %4 = call @sparseCoordinates0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %5 = builtin.unrealized_conversion_cast %4 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %6 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
    %7 = builtin.unrealized_conversion_cast %6 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    cf.br ^bb1(%c0 : index)
  ^bb1(%8: index):  // 2 preds: ^bb0, ^bb5
    %9 = builtin.unrealized_conversion_cast %8 : index to i64
    %10 = arith.cmpi slt, %8, %c3 : index
    cf.cond_br %10, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %11 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.getelementptr %11[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %13 = llvm.load %12 : !llvm.ptr -> f64
    %14 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.getelementptr %14[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %16 = llvm.load %15 : !llvm.ptr -> i64
    %17 = builtin.unrealized_conversion_cast %16 : i64 to index
    %18 = arith.addi %8, %c1 : index
    %19 = builtin.unrealized_conversion_cast %18 : index to i64
    %20 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.getelementptr %20[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %22 = llvm.load %21 : !llvm.ptr -> i64
    %23 = builtin.unrealized_conversion_cast %22 : i64 to index
    cf.br ^bb3(%17, %13 : index, f64)
  ^bb3(%24: index, %25: f64):  // 2 preds: ^bb2, ^bb4
    %26 = builtin.unrealized_conversion_cast %24 : index to i64
    %27 = arith.cmpi slt, %24, %23 : index
    cf.cond_br %27, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %28 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.getelementptr %28[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %30 = llvm.load %29 : !llvm.ptr -> i64
    %31 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.getelementptr %31[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %33 = llvm.load %32 : !llvm.ptr -> f64
    %34 = llvm.extractvalue %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.getelementptr %34[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %36 = llvm.load %35 : !llvm.ptr -> f64
    %37 = arith.mulf %33, %36 : f64
    %38 = arith.addf %25, %37 : f64
    %39 = arith.addi %24, %c1 : index
    cf.br ^bb3(%39, %38 : index, f64)
  ^bb5:  // pred: ^bb3
    %40 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.getelementptr %40[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %25, %41 : f64, !llvm.ptr
    %42 = arith.addi %8, %c1 : index
    cf.br ^bb1(%42 : index)
  ^bb6:  // pred: ^bb1
    return %arg2 : memref<3xf64>
  }
  func.func @main(%arg0: memref<3x4xf64>, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<3x4xf64> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c3 = arith.constant 3 : index
    %1 = builtin.unrealized_conversion_cast %c3 : index to i64
    %c4 = arith.constant 4 : index
    %2 = builtin.unrealized_conversion_cast %c4 : index to i64
    %c0 = arith.constant 0 : index
    %3 = builtin.unrealized_conversion_cast %c0 : index to i64
    %c1 = arith.constant 1 : index
    %4 = builtin.unrealized_conversion_cast %c1 : index to i64
    %cst = arith.constant 0.000000e+00 : f64
    %c4_i8 = arith.constant 4 : i8
    %c8_i8 = arith.constant 8 : i8
    %5 = llvm.mlir.constant(2 : index) : i64
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.alloca %5 x i8 : (i64) -> !llvm.ptr
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %7, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.mlir.constant(0 : index) : i64
    %12 = llvm.insertvalue %11, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %5, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.insertvalue %6, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = builtin.unrealized_conversion_cast %14 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xi8>
    %16 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.getelementptr %16[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %c4_i8, %17 : i8, !llvm.ptr
    %18 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.getelementptr %18[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %c8_i8, %19 : i8, !llvm.ptr
    %20 = llvm.mlir.constant(2 : index) : i64
    %21 = llvm.mlir.constant(1 : index) : i64
    %22 = llvm.alloca %20 x i64 : (i64) -> !llvm.ptr
    %23 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.insertvalue %22, %23[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.insertvalue %22, %24[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.mlir.constant(0 : index) : i64
    %27 = llvm.insertvalue %26, %25[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.insertvalue %20, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.insertvalue %21, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = builtin.unrealized_conversion_cast %29 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %31 = llvm.extractvalue %29[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.getelementptr %31[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %1, %32 : i64, !llvm.ptr
    %33 = llvm.extractvalue %29[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.getelementptr %33[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %2, %34 : i64, !llvm.ptr
    %35 = llvm.mlir.constant(2 : index) : i64
    %36 = llvm.mlir.constant(1 : index) : i64
    %37 = llvm.alloca %35 x i64 : (i64) -> !llvm.ptr
    %38 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %39 = llvm.insertvalue %37, %38[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.insertvalue %37, %39[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.mlir.constant(0 : index) : i64
    %42 = llvm.insertvalue %41, %40[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %43 = llvm.insertvalue %35, %42[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %44 = llvm.insertvalue %36, %43[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = builtin.unrealized_conversion_cast %44 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %46 = llvm.extractvalue %44[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.getelementptr %46[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %3, %47 : i64, !llvm.ptr
    %48 = llvm.extractvalue %44[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %49 = llvm.getelementptr %48[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %4, %49 : i64, !llvm.ptr
    %50 = llvm.mlir.zero : !llvm.ptr
    %51 = call @newSparseTensor(%30, %30, %15, %45, %45, %c0_i32, %c0_i32, %c1_i32, %c0_i32, %50) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    %52 = llvm.mlir.constant(2 : index) : i64
    %53 = llvm.mlir.constant(1 : index) : i64
    %54 = llvm.alloca %52 x i64 : (i64) -> !llvm.ptr
    %55 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %56 = llvm.insertvalue %54, %55[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = llvm.insertvalue %54, %56[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.mlir.constant(0 : index) : i64
    %59 = llvm.insertvalue %58, %57[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %60 = llvm.insertvalue %52, %59[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %61 = llvm.insertvalue %53, %60[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %62 = builtin.unrealized_conversion_cast %61 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %63 = llvm.mlir.constant(1 : index) : i64
    %64 = llvm.alloca %63 x f64 : (i64) -> !llvm.ptr
    %65 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %66 = llvm.insertvalue %64, %65[0] : !llvm.struct<(ptr, ptr, i64)> 
    %67 = llvm.insertvalue %64, %66[1] : !llvm.struct<(ptr, ptr, i64)> 
    %68 = llvm.mlir.constant(0 : index) : i64
    %69 = llvm.insertvalue %68, %67[2] : !llvm.struct<(ptr, ptr, i64)> 
    %70 = builtin.unrealized_conversion_cast %69 : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    cf.br ^bb1(%c0 : index)
  ^bb1(%71: index):  // 2 preds: ^bb0, ^bb7
    %72 = builtin.unrealized_conversion_cast %71 : index to i64
    %73 = arith.cmpi slt, %71, %c3 : index
    cf.cond_br %73, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%74: index):  // 2 preds: ^bb2, ^bb6
    %75 = builtin.unrealized_conversion_cast %74 : index to i64
    %76 = arith.cmpi slt, %74, %c4 : index
    cf.cond_br %76, ^bb4, ^bb7
  ^bb4:  // pred: ^bb3
    %77 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.mlir.constant(4 : index) : i64
    %79 = llvm.mul %72, %78  : i64
    %80 = llvm.add %79, %75  : i64
    %81 = llvm.getelementptr %77[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %82 = llvm.load %81 : !llvm.ptr -> f64
    %83 = arith.cmpf une, %82, %cst : f64
    cf.cond_br %83, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %84 = llvm.extractvalue %61[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %85 = llvm.getelementptr %84[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %72, %85 : i64, !llvm.ptr
    %86 = llvm.extractvalue %61[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %87 = llvm.getelementptr %86[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %75, %87 : i64, !llvm.ptr
    %88 = llvm.extractvalue %69[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %82, %88 : f64, !llvm.ptr
    call @lexInsertF64(%51, %62, %70) : (!llvm.ptr, memref<?xindex>, memref<f64>) -> ()
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %89 = arith.addi %74, %c1 : index
    cf.br ^bb3(%89 : index)
  ^bb7:  // pred: ^bb3
    %90 = arith.addi %71, %c1 : index
    cf.br ^bb1(%90 : index)
  ^bb8:  // pred: ^bb1
    call @endLexInsert(%51) : (!llvm.ptr) -> ()
    %91 = call @spMV(%51, %arg1, %arg2) : (!llvm.ptr, memref<4xf64>, memref<3xf64>) -> memref<3xf64>
    return %91 : memref<3xf64>
  }
}


// -----// IR Dump After ConvertComplexToStandard (convert-complex-to-standard) //----- //
func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertComplexToStandard (convert-complex-to-standard) //----- //
func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertComplexToStandard (convert-complex-to-standard) //----- //
func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertComplexToStandard (convert-complex-to-standard) //----- //
func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertComplexToStandard (convert-complex-to-standard) //----- //
func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertComplexToStandard (convert-complex-to-standard) //----- //
func.func private @endLexInsert(!llvm.ptr)

// -----// IR Dump After ArithExpandOpsPass (arith-expand) //----- //
func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}

// -----// IR Dump After ArithExpandOpsPass (arith-expand) //----- //
func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After ArithExpandOpsPass (arith-expand) //----- //
func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}

// -----// IR Dump After ArithExpandOpsPass (arith-expand) //----- //
func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertComplexToStandard (convert-complex-to-standard) //----- //
func.func @spMV(%arg0: !llvm.ptr, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> {
  %0 = builtin.unrealized_conversion_cast %arg2 : memref<3xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %1 = builtin.unrealized_conversion_cast %arg1 : memref<4xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %2 = call @sparsePositions0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
  %3 = builtin.unrealized_conversion_cast %2 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %4 = call @sparseCoordinates0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
  %5 = builtin.unrealized_conversion_cast %4 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %6 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
  %7 = builtin.unrealized_conversion_cast %6 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  cf.br ^bb1(%c0 : index)
^bb1(%8: index):  // 2 preds: ^bb0, ^bb5
  %9 = builtin.unrealized_conversion_cast %8 : index to i64
  %10 = arith.cmpi slt, %8, %c3 : index
  cf.cond_br %10, ^bb2, ^bb6
^bb2:  // pred: ^bb1
  %11 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %12 = llvm.getelementptr %11[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %13 = llvm.load %12 : !llvm.ptr -> f64
  %14 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %15 = llvm.getelementptr %14[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  %16 = llvm.load %15 : !llvm.ptr -> i64
  %17 = builtin.unrealized_conversion_cast %16 : i64 to index
  %18 = arith.addi %8, %c1 : index
  %19 = builtin.unrealized_conversion_cast %18 : index to i64
  %20 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %21 = llvm.getelementptr %20[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  %22 = llvm.load %21 : !llvm.ptr -> i64
  %23 = builtin.unrealized_conversion_cast %22 : i64 to index
  cf.br ^bb3(%17, %13 : index, f64)
^bb3(%24: index, %25: f64):  // 2 preds: ^bb2, ^bb4
  %26 = builtin.unrealized_conversion_cast %24 : index to i64
  %27 = arith.cmpi slt, %24, %23 : index
  cf.cond_br %27, ^bb4, ^bb5
^bb4:  // pred: ^bb3
  %28 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %29 = llvm.getelementptr %28[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  %30 = llvm.load %29 : !llvm.ptr -> i64
  %31 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %32 = llvm.getelementptr %31[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %33 = llvm.load %32 : !llvm.ptr -> f64
  %34 = llvm.extractvalue %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %35 = llvm.getelementptr %34[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %36 = llvm.load %35 : !llvm.ptr -> f64
  %37 = arith.mulf %33, %36 : f64
  %38 = arith.addf %25, %37 : f64
  %39 = arith.addi %24, %c1 : index
  cf.br ^bb3(%39, %38 : index, f64)
^bb5:  // pred: ^bb3
  %40 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %41 = llvm.getelementptr %40[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %25, %41 : f64, !llvm.ptr
  %42 = arith.addi %8, %c1 : index
  cf.br ^bb1(%42 : index)
^bb6:  // pred: ^bb1
  return %arg2 : memref<3xf64>
}

// -----// IR Dump After ArithExpandOpsPass (arith-expand) //----- //
func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertComplexToStandard (convert-complex-to-standard) //----- //
func.func @main(%arg0: memref<3x4xf64>, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
  %0 = builtin.unrealized_conversion_cast %arg0 : memref<3x4xf64> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c3 = arith.constant 3 : index
  %1 = builtin.unrealized_conversion_cast %c3 : index to i64
  %c4 = arith.constant 4 : index
  %2 = builtin.unrealized_conversion_cast %c4 : index to i64
  %c0 = arith.constant 0 : index
  %3 = builtin.unrealized_conversion_cast %c0 : index to i64
  %c1 = arith.constant 1 : index
  %4 = builtin.unrealized_conversion_cast %c1 : index to i64
  %cst = arith.constant 0.000000e+00 : f64
  %c4_i8 = arith.constant 4 : i8
  %c8_i8 = arith.constant 8 : i8
  %5 = llvm.mlir.constant(2 : index) : i64
  %6 = llvm.mlir.constant(1 : index) : i64
  %7 = llvm.alloca %5 x i8 : (i64) -> !llvm.ptr
  %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %10 = llvm.insertvalue %7, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %11 = llvm.mlir.constant(0 : index) : i64
  %12 = llvm.insertvalue %11, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %13 = llvm.insertvalue %5, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %14 = llvm.insertvalue %6, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %15 = builtin.unrealized_conversion_cast %14 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xi8>
  %16 = llvm.getelementptr %7[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %c4_i8, %16 : i8, !llvm.ptr
  %17 = llvm.getelementptr %7[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %c8_i8, %17 : i8, !llvm.ptr
  %18 = llvm.mlir.constant(2 : index) : i64
  %19 = llvm.mlir.constant(1 : index) : i64
  %20 = llvm.alloca %18 x i64 : (i64) -> !llvm.ptr
  %21 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %22 = llvm.insertvalue %20, %21[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %23 = llvm.insertvalue %20, %22[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %24 = llvm.mlir.constant(0 : index) : i64
  %25 = llvm.insertvalue %24, %23[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %26 = llvm.insertvalue %18, %25[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %27 = llvm.insertvalue %19, %26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %28 = builtin.unrealized_conversion_cast %27 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
  %29 = llvm.getelementptr %20[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %1, %29 : i64, !llvm.ptr
  %30 = llvm.getelementptr %20[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %2, %30 : i64, !llvm.ptr
  %31 = llvm.mlir.constant(2 : index) : i64
  %32 = llvm.mlir.constant(1 : index) : i64
  %33 = llvm.alloca %31 x i64 : (i64) -> !llvm.ptr
  %34 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %35 = llvm.insertvalue %33, %34[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %36 = llvm.insertvalue %33, %35[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %37 = llvm.mlir.constant(0 : index) : i64
  %38 = llvm.insertvalue %37, %36[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %39 = llvm.insertvalue %31, %38[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %40 = llvm.insertvalue %32, %39[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %41 = builtin.unrealized_conversion_cast %40 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
  %42 = llvm.getelementptr %33[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %3, %42 : i64, !llvm.ptr
  %43 = llvm.getelementptr %33[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %4, %43 : i64, !llvm.ptr
  %44 = llvm.mlir.zero : !llvm.ptr
  %45 = call @newSparseTensor(%28, %28, %15, %41, %41, %c0_i32, %c0_i32, %c1_i32, %c0_i32, %44) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
  %46 = llvm.mlir.constant(2 : index) : i64
  %47 = llvm.mlir.constant(1 : index) : i64
  %48 = llvm.alloca %46 x i64 : (i64) -> !llvm.ptr
  %49 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %50 = llvm.insertvalue %48, %49[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %51 = llvm.insertvalue %48, %50[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %52 = llvm.mlir.constant(0 : index) : i64
  %53 = llvm.insertvalue %52, %51[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %54 = llvm.insertvalue %46, %53[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %55 = llvm.insertvalue %47, %54[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %56 = builtin.unrealized_conversion_cast %55 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
  %57 = llvm.mlir.constant(1 : index) : i64
  %58 = llvm.alloca %57 x f64 : (i64) -> !llvm.ptr
  %59 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
  %60 = llvm.insertvalue %58, %59[0] : !llvm.struct<(ptr, ptr, i64)> 
  %61 = llvm.insertvalue %58, %60[1] : !llvm.struct<(ptr, ptr, i64)> 
  %62 = llvm.mlir.constant(0 : index) : i64
  %63 = llvm.insertvalue %62, %61[2] : !llvm.struct<(ptr, ptr, i64)> 
  %64 = builtin.unrealized_conversion_cast %63 : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
  cf.br ^bb1(%c0 : index)
^bb1(%65: index):  // 2 preds: ^bb0, ^bb7
  %66 = builtin.unrealized_conversion_cast %65 : index to i64
  %67 = arith.cmpi slt, %65, %c3 : index
  cf.cond_br %67, ^bb2, ^bb8
^bb2:  // pred: ^bb1
  cf.br ^bb3(%c0 : index)
^bb3(%68: index):  // 2 preds: ^bb2, ^bb6
  %69 = builtin.unrealized_conversion_cast %68 : index to i64
  %70 = arith.cmpi slt, %68, %c4 : index
  cf.cond_br %70, ^bb4, ^bb7
^bb4:  // pred: ^bb3
  %71 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
  %72 = llvm.mlir.constant(4 : index) : i64
  %73 = llvm.mul %66, %72  : i64
  %74 = llvm.add %73, %69  : i64
  %75 = llvm.getelementptr %71[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %76 = llvm.load %75 : !llvm.ptr -> f64
  %77 = arith.cmpf une, %76, %cst : f64
  cf.cond_br %77, ^bb5, ^bb6
^bb5:  // pred: ^bb4
  %78 = llvm.getelementptr %48[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %66, %78 : i64, !llvm.ptr
  %79 = llvm.getelementptr %48[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %69, %79 : i64, !llvm.ptr
  llvm.store %76, %58 : f64, !llvm.ptr
  call @lexInsertF64(%45, %56, %64) : (!llvm.ptr, memref<?xindex>, memref<f64>) -> ()
  cf.br ^bb6
^bb6:  // 2 preds: ^bb4, ^bb5
  %80 = arith.addi %68, %c1 : index
  cf.br ^bb3(%80 : index)
^bb7:  // pred: ^bb3
  %81 = arith.addi %65, %c1 : index
  cf.br ^bb1(%81 : index)
^bb8:  // pred: ^bb1
  call @endLexInsert(%45) : (!llvm.ptr) -> ()
  %82 = call @spMV(%45, %arg1, %arg2) : (!llvm.ptr, memref<4xf64>, memref<3xf64>) -> memref<3xf64>
  return %82 : memref<3xf64>
}

// -----// IR Dump After ArithExpandOpsPass (arith-expand) //----- //
func.func private @endLexInsert(!llvm.ptr)

// -----// IR Dump After ConvertMathToLLVMPass (convert-math-to-llvm) //----- //
func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertMathToLLVMPass (convert-math-to-llvm) //----- //
func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertMathToLLVMPass (convert-math-to-llvm) //----- //
func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertMathToLLVMPass (convert-math-to-llvm) //----- //
func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertMathToLLVMPass (convert-math-to-llvm) //----- //
func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}

// -----// IR Dump After ConvertMathToLLVMPass (convert-math-to-llvm) //----- //
func.func private @endLexInsert(!llvm.ptr)

// -----// IR Dump After ArithExpandOpsPass (arith-expand) //----- //
func.func @spMV(%arg0: !llvm.ptr, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> {
  %0 = builtin.unrealized_conversion_cast %arg2 : memref<3xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %1 = builtin.unrealized_conversion_cast %arg1 : memref<4xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %2 = call @sparsePositions0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
  %3 = builtin.unrealized_conversion_cast %2 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %4 = call @sparseCoordinates0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
  %5 = builtin.unrealized_conversion_cast %4 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %6 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
  %7 = builtin.unrealized_conversion_cast %6 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  cf.br ^bb1(%c0 : index)
^bb1(%8: index):  // 2 preds: ^bb0, ^bb5
  %9 = builtin.unrealized_conversion_cast %8 : index to i64
  %10 = arith.cmpi slt, %8, %c3 : index
  cf.cond_br %10, ^bb2, ^bb6
^bb2:  // pred: ^bb1
  %11 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %12 = llvm.getelementptr %11[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %13 = llvm.load %12 : !llvm.ptr -> f64
  %14 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %15 = llvm.getelementptr %14[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  %16 = llvm.load %15 : !llvm.ptr -> i64
  %17 = builtin.unrealized_conversion_cast %16 : i64 to index
  %18 = arith.addi %8, %c1 : index
  %19 = builtin.unrealized_conversion_cast %18 : index to i64
  %20 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %21 = llvm.getelementptr %20[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  %22 = llvm.load %21 : !llvm.ptr -> i64
  %23 = builtin.unrealized_conversion_cast %22 : i64 to index
  cf.br ^bb3(%17, %13 : index, f64)
^bb3(%24: index, %25: f64):  // 2 preds: ^bb2, ^bb4
  %26 = builtin.unrealized_conversion_cast %24 : index to i64
  %27 = arith.cmpi slt, %24, %23 : index
  cf.cond_br %27, ^bb4, ^bb5
^bb4:  // pred: ^bb3
  %28 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %29 = llvm.getelementptr %28[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  %30 = llvm.load %29 : !llvm.ptr -> i64
  %31 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %32 = llvm.getelementptr %31[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %33 = llvm.load %32 : !llvm.ptr -> f64
  %34 = llvm.extractvalue %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %35 = llvm.getelementptr %34[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %36 = llvm.load %35 : !llvm.ptr -> f64
  %37 = arith.mulf %33, %36 : f64
  %38 = arith.addf %25, %37 : f64
  %39 = arith.addi %24, %c1 : index
  cf.br ^bb3(%39, %38 : index, f64)
^bb5:  // pred: ^bb3
  %40 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %41 = llvm.getelementptr %40[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %25, %41 : f64, !llvm.ptr
  %42 = arith.addi %8, %c1 : index
  cf.br ^bb1(%42 : index)
^bb6:  // pred: ^bb1
  return %arg2 : memref<3xf64>
}

// -----// IR Dump After ArithExpandOpsPass (arith-expand) //----- //
func.func @main(%arg0: memref<3x4xf64>, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
  %0 = builtin.unrealized_conversion_cast %arg0 : memref<3x4xf64> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c3 = arith.constant 3 : index
  %1 = builtin.unrealized_conversion_cast %c3 : index to i64
  %c4 = arith.constant 4 : index
  %2 = builtin.unrealized_conversion_cast %c4 : index to i64
  %c0 = arith.constant 0 : index
  %3 = builtin.unrealized_conversion_cast %c0 : index to i64
  %c1 = arith.constant 1 : index
  %4 = builtin.unrealized_conversion_cast %c1 : index to i64
  %cst = arith.constant 0.000000e+00 : f64
  %c4_i8 = arith.constant 4 : i8
  %c8_i8 = arith.constant 8 : i8
  %5 = llvm.mlir.constant(2 : index) : i64
  %6 = llvm.mlir.constant(1 : index) : i64
  %7 = llvm.alloca %5 x i8 : (i64) -> !llvm.ptr
  %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %10 = llvm.insertvalue %7, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %11 = llvm.mlir.constant(0 : index) : i64
  %12 = llvm.insertvalue %11, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %13 = llvm.insertvalue %5, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %14 = llvm.insertvalue %6, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %15 = builtin.unrealized_conversion_cast %14 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xi8>
  %16 = llvm.getelementptr %7[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %c4_i8, %16 : i8, !llvm.ptr
  %17 = llvm.getelementptr %7[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %c8_i8, %17 : i8, !llvm.ptr
  %18 = llvm.mlir.constant(2 : index) : i64
  %19 = llvm.mlir.constant(1 : index) : i64
  %20 = llvm.alloca %18 x i64 : (i64) -> !llvm.ptr
  %21 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %22 = llvm.insertvalue %20, %21[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %23 = llvm.insertvalue %20, %22[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %24 = llvm.mlir.constant(0 : index) : i64
  %25 = llvm.insertvalue %24, %23[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %26 = llvm.insertvalue %18, %25[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %27 = llvm.insertvalue %19, %26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %28 = builtin.unrealized_conversion_cast %27 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
  %29 = llvm.getelementptr %20[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %1, %29 : i64, !llvm.ptr
  %30 = llvm.getelementptr %20[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %2, %30 : i64, !llvm.ptr
  %31 = llvm.mlir.constant(2 : index) : i64
  %32 = llvm.mlir.constant(1 : index) : i64
  %33 = llvm.alloca %31 x i64 : (i64) -> !llvm.ptr
  %34 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %35 = llvm.insertvalue %33, %34[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %36 = llvm.insertvalue %33, %35[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %37 = llvm.mlir.constant(0 : index) : i64
  %38 = llvm.insertvalue %37, %36[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %39 = llvm.insertvalue %31, %38[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %40 = llvm.insertvalue %32, %39[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %41 = builtin.unrealized_conversion_cast %40 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
  %42 = llvm.getelementptr %33[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %3, %42 : i64, !llvm.ptr
  %43 = llvm.getelementptr %33[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %4, %43 : i64, !llvm.ptr
  %44 = llvm.mlir.zero : !llvm.ptr
  %45 = call @newSparseTensor(%28, %28, %15, %41, %41, %c0_i32, %c0_i32, %c1_i32, %c0_i32, %44) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
  %46 = llvm.mlir.constant(2 : index) : i64
  %47 = llvm.mlir.constant(1 : index) : i64
  %48 = llvm.alloca %46 x i64 : (i64) -> !llvm.ptr
  %49 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %50 = llvm.insertvalue %48, %49[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %51 = llvm.insertvalue %48, %50[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %52 = llvm.mlir.constant(0 : index) : i64
  %53 = llvm.insertvalue %52, %51[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %54 = llvm.insertvalue %46, %53[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %55 = llvm.insertvalue %47, %54[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %56 = builtin.unrealized_conversion_cast %55 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
  %57 = llvm.mlir.constant(1 : index) : i64
  %58 = llvm.alloca %57 x f64 : (i64) -> !llvm.ptr
  %59 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
  %60 = llvm.insertvalue %58, %59[0] : !llvm.struct<(ptr, ptr, i64)> 
  %61 = llvm.insertvalue %58, %60[1] : !llvm.struct<(ptr, ptr, i64)> 
  %62 = llvm.mlir.constant(0 : index) : i64
  %63 = llvm.insertvalue %62, %61[2] : !llvm.struct<(ptr, ptr, i64)> 
  %64 = builtin.unrealized_conversion_cast %63 : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
  cf.br ^bb1(%c0 : index)
^bb1(%65: index):  // 2 preds: ^bb0, ^bb7
  %66 = builtin.unrealized_conversion_cast %65 : index to i64
  %67 = arith.cmpi slt, %65, %c3 : index
  cf.cond_br %67, ^bb2, ^bb8
^bb2:  // pred: ^bb1
  cf.br ^bb3(%c0 : index)
^bb3(%68: index):  // 2 preds: ^bb2, ^bb6
  %69 = builtin.unrealized_conversion_cast %68 : index to i64
  %70 = arith.cmpi slt, %68, %c4 : index
  cf.cond_br %70, ^bb4, ^bb7
^bb4:  // pred: ^bb3
  %71 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
  %72 = llvm.mlir.constant(4 : index) : i64
  %73 = llvm.mul %66, %72  : i64
  %74 = llvm.add %73, %69  : i64
  %75 = llvm.getelementptr %71[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %76 = llvm.load %75 : !llvm.ptr -> f64
  %77 = arith.cmpf une, %76, %cst : f64
  cf.cond_br %77, ^bb5, ^bb6
^bb5:  // pred: ^bb4
  %78 = llvm.getelementptr %48[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %66, %78 : i64, !llvm.ptr
  %79 = llvm.getelementptr %48[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %69, %79 : i64, !llvm.ptr
  llvm.store %76, %58 : f64, !llvm.ptr
  call @lexInsertF64(%45, %56, %64) : (!llvm.ptr, memref<?xindex>, memref<f64>) -> ()
  cf.br ^bb6
^bb6:  // 2 preds: ^bb4, ^bb5
  %80 = arith.addi %68, %c1 : index
  cf.br ^bb3(%80 : index)
^bb7:  // pred: ^bb3
  %81 = arith.addi %65, %c1 : index
  cf.br ^bb1(%81 : index)
^bb8:  // pred: ^bb1
  call @endLexInsert(%45) : (!llvm.ptr) -> ()
  %82 = call @spMV(%45, %arg1, %arg2) : (!llvm.ptr, memref<4xf64>, memref<3xf64>) -> memref<3xf64>
  return %82 : memref<3xf64>
}

// -----// IR Dump After ConvertMathToLLVMPass (convert-math-to-llvm) //----- //
func.func @spMV(%arg0: !llvm.ptr, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> {
  %0 = builtin.unrealized_conversion_cast %arg2 : memref<3xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %1 = builtin.unrealized_conversion_cast %arg1 : memref<4xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %2 = call @sparsePositions0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
  %3 = builtin.unrealized_conversion_cast %2 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %4 = call @sparseCoordinates0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
  %5 = builtin.unrealized_conversion_cast %4 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %6 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
  %7 = builtin.unrealized_conversion_cast %6 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  cf.br ^bb1(%c0 : index)
^bb1(%8: index):  // 2 preds: ^bb0, ^bb5
  %9 = builtin.unrealized_conversion_cast %8 : index to i64
  %10 = arith.cmpi slt, %8, %c3 : index
  cf.cond_br %10, ^bb2, ^bb6
^bb2:  // pred: ^bb1
  %11 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %12 = llvm.getelementptr %11[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %13 = llvm.load %12 : !llvm.ptr -> f64
  %14 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %15 = llvm.getelementptr %14[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  %16 = llvm.load %15 : !llvm.ptr -> i64
  %17 = builtin.unrealized_conversion_cast %16 : i64 to index
  %18 = arith.addi %8, %c1 : index
  %19 = builtin.unrealized_conversion_cast %18 : index to i64
  %20 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %21 = llvm.getelementptr %20[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  %22 = llvm.load %21 : !llvm.ptr -> i64
  %23 = builtin.unrealized_conversion_cast %22 : i64 to index
  cf.br ^bb3(%17, %13 : index, f64)
^bb3(%24: index, %25: f64):  // 2 preds: ^bb2, ^bb4
  %26 = builtin.unrealized_conversion_cast %24 : index to i64
  %27 = arith.cmpi slt, %24, %23 : index
  cf.cond_br %27, ^bb4, ^bb5
^bb4:  // pred: ^bb3
  %28 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %29 = llvm.getelementptr %28[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  %30 = llvm.load %29 : !llvm.ptr -> i64
  %31 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %32 = llvm.getelementptr %31[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %33 = llvm.load %32 : !llvm.ptr -> f64
  %34 = llvm.extractvalue %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %35 = llvm.getelementptr %34[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %36 = llvm.load %35 : !llvm.ptr -> f64
  %37 = arith.mulf %33, %36 : f64
  %38 = arith.addf %25, %37 : f64
  %39 = arith.addi %24, %c1 : index
  cf.br ^bb3(%39, %38 : index, f64)
^bb5:  // pred: ^bb3
  %40 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %41 = llvm.getelementptr %40[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %25, %41 : f64, !llvm.ptr
  %42 = arith.addi %8, %c1 : index
  cf.br ^bb1(%42 : index)
^bb6:  // pred: ^bb1
  return %arg2 : memref<3xf64>
}

// -----// IR Dump After ConvertMathToLLVMPass (convert-math-to-llvm) //----- //
func.func @main(%arg0: memref<3x4xf64>, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
  %0 = builtin.unrealized_conversion_cast %arg0 : memref<3x4xf64> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c3 = arith.constant 3 : index
  %1 = builtin.unrealized_conversion_cast %c3 : index to i64
  %c4 = arith.constant 4 : index
  %2 = builtin.unrealized_conversion_cast %c4 : index to i64
  %c0 = arith.constant 0 : index
  %3 = builtin.unrealized_conversion_cast %c0 : index to i64
  %c1 = arith.constant 1 : index
  %4 = builtin.unrealized_conversion_cast %c1 : index to i64
  %cst = arith.constant 0.000000e+00 : f64
  %c4_i8 = arith.constant 4 : i8
  %c8_i8 = arith.constant 8 : i8
  %5 = llvm.mlir.constant(2 : index) : i64
  %6 = llvm.mlir.constant(1 : index) : i64
  %7 = llvm.alloca %5 x i8 : (i64) -> !llvm.ptr
  %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %10 = llvm.insertvalue %7, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %11 = llvm.mlir.constant(0 : index) : i64
  %12 = llvm.insertvalue %11, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %13 = llvm.insertvalue %5, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %14 = llvm.insertvalue %6, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %15 = builtin.unrealized_conversion_cast %14 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xi8>
  %16 = llvm.getelementptr %7[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %c4_i8, %16 : i8, !llvm.ptr
  %17 = llvm.getelementptr %7[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %c8_i8, %17 : i8, !llvm.ptr
  %18 = llvm.mlir.constant(2 : index) : i64
  %19 = llvm.mlir.constant(1 : index) : i64
  %20 = llvm.alloca %18 x i64 : (i64) -> !llvm.ptr
  %21 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %22 = llvm.insertvalue %20, %21[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %23 = llvm.insertvalue %20, %22[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %24 = llvm.mlir.constant(0 : index) : i64
  %25 = llvm.insertvalue %24, %23[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %26 = llvm.insertvalue %18, %25[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %27 = llvm.insertvalue %19, %26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %28 = builtin.unrealized_conversion_cast %27 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
  %29 = llvm.getelementptr %20[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %1, %29 : i64, !llvm.ptr
  %30 = llvm.getelementptr %20[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %2, %30 : i64, !llvm.ptr
  %31 = llvm.mlir.constant(2 : index) : i64
  %32 = llvm.mlir.constant(1 : index) : i64
  %33 = llvm.alloca %31 x i64 : (i64) -> !llvm.ptr
  %34 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %35 = llvm.insertvalue %33, %34[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %36 = llvm.insertvalue %33, %35[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %37 = llvm.mlir.constant(0 : index) : i64
  %38 = llvm.insertvalue %37, %36[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %39 = llvm.insertvalue %31, %38[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %40 = llvm.insertvalue %32, %39[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %41 = builtin.unrealized_conversion_cast %40 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
  %42 = llvm.getelementptr %33[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %3, %42 : i64, !llvm.ptr
  %43 = llvm.getelementptr %33[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %4, %43 : i64, !llvm.ptr
  %44 = llvm.mlir.zero : !llvm.ptr
  %45 = call @newSparseTensor(%28, %28, %15, %41, %41, %c0_i32, %c0_i32, %c1_i32, %c0_i32, %44) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
  %46 = llvm.mlir.constant(2 : index) : i64
  %47 = llvm.mlir.constant(1 : index) : i64
  %48 = llvm.alloca %46 x i64 : (i64) -> !llvm.ptr
  %49 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %50 = llvm.insertvalue %48, %49[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %51 = llvm.insertvalue %48, %50[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %52 = llvm.mlir.constant(0 : index) : i64
  %53 = llvm.insertvalue %52, %51[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %54 = llvm.insertvalue %46, %53[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %55 = llvm.insertvalue %47, %54[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  %56 = builtin.unrealized_conversion_cast %55 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
  %57 = llvm.mlir.constant(1 : index) : i64
  %58 = llvm.alloca %57 x f64 : (i64) -> !llvm.ptr
  %59 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
  %60 = llvm.insertvalue %58, %59[0] : !llvm.struct<(ptr, ptr, i64)> 
  %61 = llvm.insertvalue %58, %60[1] : !llvm.struct<(ptr, ptr, i64)> 
  %62 = llvm.mlir.constant(0 : index) : i64
  %63 = llvm.insertvalue %62, %61[2] : !llvm.struct<(ptr, ptr, i64)> 
  %64 = builtin.unrealized_conversion_cast %63 : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
  cf.br ^bb1(%c0 : index)
^bb1(%65: index):  // 2 preds: ^bb0, ^bb7
  %66 = builtin.unrealized_conversion_cast %65 : index to i64
  %67 = arith.cmpi slt, %65, %c3 : index
  cf.cond_br %67, ^bb2, ^bb8
^bb2:  // pred: ^bb1
  cf.br ^bb3(%c0 : index)
^bb3(%68: index):  // 2 preds: ^bb2, ^bb6
  %69 = builtin.unrealized_conversion_cast %68 : index to i64
  %70 = arith.cmpi slt, %68, %c4 : index
  cf.cond_br %70, ^bb4, ^bb7
^bb4:  // pred: ^bb3
  %71 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
  %72 = llvm.mlir.constant(4 : index) : i64
  %73 = llvm.mul %66, %72  : i64
  %74 = llvm.add %73, %69  : i64
  %75 = llvm.getelementptr %71[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %76 = llvm.load %75 : !llvm.ptr -> f64
  %77 = arith.cmpf une, %76, %cst : f64
  cf.cond_br %77, ^bb5, ^bb6
^bb5:  // pred: ^bb4
  %78 = llvm.getelementptr %48[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %66, %78 : i64, !llvm.ptr
  %79 = llvm.getelementptr %48[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store %69, %79 : i64, !llvm.ptr
  llvm.store %76, %58 : f64, !llvm.ptr
  call @lexInsertF64(%45, %56, %64) : (!llvm.ptr, memref<?xindex>, memref<f64>) -> ()
  cf.br ^bb6
^bb6:  // 2 preds: ^bb4, ^bb5
  %80 = arith.addi %68, %c1 : index
  cf.br ^bb3(%80 : index)
^bb7:  // pred: ^bb3
  %81 = arith.addi %65, %c1 : index
  cf.br ^bb1(%81 : index)
^bb8:  // pred: ^bb1
  call @endLexInsert(%45) : (!llvm.ptr) -> ()
  %82 = call @spMV(%45, %arg1, %arg2) : (!llvm.ptr, memref<4xf64>, memref<3xf64>) -> memref<3xf64>
  return %82 : memref<3xf64>
}

// -----// IR Dump After ConvertMathToLibm (convert-math-to-libm) //----- //
module {
  func.func private @endLexInsert(!llvm.ptr)
  func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}
  func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
  func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}
  func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func @spMV(%arg0: !llvm.ptr, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> {
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<3xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<4xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %2 = call @sparsePositions0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %3 = builtin.unrealized_conversion_cast %2 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %4 = call @sparseCoordinates0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %5 = builtin.unrealized_conversion_cast %4 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %6 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
    %7 = builtin.unrealized_conversion_cast %6 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    cf.br ^bb1(%c0 : index)
  ^bb1(%8: index):  // 2 preds: ^bb0, ^bb5
    %9 = builtin.unrealized_conversion_cast %8 : index to i64
    %10 = arith.cmpi slt, %8, %c3 : index
    cf.cond_br %10, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %11 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.getelementptr %11[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %13 = llvm.load %12 : !llvm.ptr -> f64
    %14 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.getelementptr %14[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %16 = llvm.load %15 : !llvm.ptr -> i64
    %17 = builtin.unrealized_conversion_cast %16 : i64 to index
    %18 = arith.addi %8, %c1 : index
    %19 = builtin.unrealized_conversion_cast %18 : index to i64
    %20 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.getelementptr %20[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %22 = llvm.load %21 : !llvm.ptr -> i64
    %23 = builtin.unrealized_conversion_cast %22 : i64 to index
    cf.br ^bb3(%17, %13 : index, f64)
  ^bb3(%24: index, %25: f64):  // 2 preds: ^bb2, ^bb4
    %26 = builtin.unrealized_conversion_cast %24 : index to i64
    %27 = arith.cmpi slt, %24, %23 : index
    cf.cond_br %27, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %28 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.getelementptr %28[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %30 = llvm.load %29 : !llvm.ptr -> i64
    %31 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.getelementptr %31[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %33 = llvm.load %32 : !llvm.ptr -> f64
    %34 = llvm.extractvalue %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.getelementptr %34[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %36 = llvm.load %35 : !llvm.ptr -> f64
    %37 = arith.mulf %33, %36 : f64
    %38 = arith.addf %25, %37 : f64
    %39 = arith.addi %24, %c1 : index
    cf.br ^bb3(%39, %38 : index, f64)
  ^bb5:  // pred: ^bb3
    %40 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.getelementptr %40[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %25, %41 : f64, !llvm.ptr
    %42 = arith.addi %8, %c1 : index
    cf.br ^bb1(%42 : index)
  ^bb6:  // pred: ^bb1
    return %arg2 : memref<3xf64>
  }
  func.func @main(%arg0: memref<3x4xf64>, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<3x4xf64> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c3 = arith.constant 3 : index
    %1 = builtin.unrealized_conversion_cast %c3 : index to i64
    %c4 = arith.constant 4 : index
    %2 = builtin.unrealized_conversion_cast %c4 : index to i64
    %c0 = arith.constant 0 : index
    %3 = builtin.unrealized_conversion_cast %c0 : index to i64
    %c1 = arith.constant 1 : index
    %4 = builtin.unrealized_conversion_cast %c1 : index to i64
    %cst = arith.constant 0.000000e+00 : f64
    %c4_i8 = arith.constant 4 : i8
    %c8_i8 = arith.constant 8 : i8
    %5 = llvm.mlir.constant(2 : index) : i64
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.alloca %5 x i8 : (i64) -> !llvm.ptr
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %7, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.mlir.constant(0 : index) : i64
    %12 = llvm.insertvalue %11, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %5, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.insertvalue %6, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = builtin.unrealized_conversion_cast %14 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xi8>
    %16 = llvm.getelementptr %7[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %c4_i8, %16 : i8, !llvm.ptr
    %17 = llvm.getelementptr %7[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %c8_i8, %17 : i8, !llvm.ptr
    %18 = llvm.mlir.constant(2 : index) : i64
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.alloca %18 x i64 : (i64) -> !llvm.ptr
    %21 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.insertvalue %20, %21[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.insertvalue %20, %22[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.mlir.constant(0 : index) : i64
    %25 = llvm.insertvalue %24, %23[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.insertvalue %18, %25[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.insertvalue %19, %26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = builtin.unrealized_conversion_cast %27 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %29 = llvm.getelementptr %20[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %1, %29 : i64, !llvm.ptr
    %30 = llvm.getelementptr %20[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %2, %30 : i64, !llvm.ptr
    %31 = llvm.mlir.constant(2 : index) : i64
    %32 = llvm.mlir.constant(1 : index) : i64
    %33 = llvm.alloca %31 x i64 : (i64) -> !llvm.ptr
    %34 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %35 = llvm.insertvalue %33, %34[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.insertvalue %33, %35[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.mlir.constant(0 : index) : i64
    %38 = llvm.insertvalue %37, %36[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %39 = llvm.insertvalue %31, %38[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.insertvalue %32, %39[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = builtin.unrealized_conversion_cast %40 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %42 = llvm.getelementptr %33[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %3, %42 : i64, !llvm.ptr
    %43 = llvm.getelementptr %33[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %4, %43 : i64, !llvm.ptr
    %44 = llvm.mlir.zero : !llvm.ptr
    %45 = call @newSparseTensor(%28, %28, %15, %41, %41, %c0_i32, %c0_i32, %c1_i32, %c0_i32, %44) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    %46 = llvm.mlir.constant(2 : index) : i64
    %47 = llvm.mlir.constant(1 : index) : i64
    %48 = llvm.alloca %46 x i64 : (i64) -> !llvm.ptr
    %49 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %50 = llvm.insertvalue %48, %49[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %51 = llvm.insertvalue %48, %50[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %52 = llvm.mlir.constant(0 : index) : i64
    %53 = llvm.insertvalue %52, %51[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %54 = llvm.insertvalue %46, %53[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.insertvalue %47, %54[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = builtin.unrealized_conversion_cast %55 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %57 = llvm.mlir.constant(1 : index) : i64
    %58 = llvm.alloca %57 x f64 : (i64) -> !llvm.ptr
    %59 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %60 = llvm.insertvalue %58, %59[0] : !llvm.struct<(ptr, ptr, i64)> 
    %61 = llvm.insertvalue %58, %60[1] : !llvm.struct<(ptr, ptr, i64)> 
    %62 = llvm.mlir.constant(0 : index) : i64
    %63 = llvm.insertvalue %62, %61[2] : !llvm.struct<(ptr, ptr, i64)> 
    %64 = builtin.unrealized_conversion_cast %63 : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    cf.br ^bb1(%c0 : index)
  ^bb1(%65: index):  // 2 preds: ^bb0, ^bb7
    %66 = builtin.unrealized_conversion_cast %65 : index to i64
    %67 = arith.cmpi slt, %65, %c3 : index
    cf.cond_br %67, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%68: index):  // 2 preds: ^bb2, ^bb6
    %69 = builtin.unrealized_conversion_cast %68 : index to i64
    %70 = arith.cmpi slt, %68, %c4 : index
    cf.cond_br %70, ^bb4, ^bb7
  ^bb4:  // pred: ^bb3
    %71 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.mlir.constant(4 : index) : i64
    %73 = llvm.mul %66, %72  : i64
    %74 = llvm.add %73, %69  : i64
    %75 = llvm.getelementptr %71[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %76 = llvm.load %75 : !llvm.ptr -> f64
    %77 = arith.cmpf une, %76, %cst : f64
    cf.cond_br %77, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %78 = llvm.getelementptr %48[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %66, %78 : i64, !llvm.ptr
    %79 = llvm.getelementptr %48[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %69, %79 : i64, !llvm.ptr
    llvm.store %76, %58 : f64, !llvm.ptr
    call @lexInsertF64(%45, %56, %64) : (!llvm.ptr, memref<?xindex>, memref<f64>) -> ()
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %80 = arith.addi %68, %c1 : index
    cf.br ^bb3(%80 : index)
  ^bb7:  // pred: ^bb3
    %81 = arith.addi %65, %c1 : index
    cf.br ^bb1(%81 : index)
  ^bb8:  // pred: ^bb1
    call @endLexInsert(%45) : (!llvm.ptr) -> ()
    %82 = call @spMV(%45, %arg1, %arg2) : (!llvm.ptr, memref<4xf64>, memref<3xf64>) -> memref<3xf64>
    return %82 : memref<3xf64>
  }
}


// -----// IR Dump After ConvertComplexToLibm (convert-complex-to-libm) //----- //
module {
  func.func private @endLexInsert(!llvm.ptr)
  func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}
  func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
  func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}
  func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func @spMV(%arg0: !llvm.ptr, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> {
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<3xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<4xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %2 = call @sparsePositions0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %3 = builtin.unrealized_conversion_cast %2 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %4 = call @sparseCoordinates0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %5 = builtin.unrealized_conversion_cast %4 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %6 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
    %7 = builtin.unrealized_conversion_cast %6 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    cf.br ^bb1(%c0 : index)
  ^bb1(%8: index):  // 2 preds: ^bb0, ^bb5
    %9 = builtin.unrealized_conversion_cast %8 : index to i64
    %10 = arith.cmpi slt, %8, %c3 : index
    cf.cond_br %10, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %11 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.getelementptr %11[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %13 = llvm.load %12 : !llvm.ptr -> f64
    %14 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.getelementptr %14[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %16 = llvm.load %15 : !llvm.ptr -> i64
    %17 = builtin.unrealized_conversion_cast %16 : i64 to index
    %18 = arith.addi %8, %c1 : index
    %19 = builtin.unrealized_conversion_cast %18 : index to i64
    %20 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.getelementptr %20[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %22 = llvm.load %21 : !llvm.ptr -> i64
    %23 = builtin.unrealized_conversion_cast %22 : i64 to index
    cf.br ^bb3(%17, %13 : index, f64)
  ^bb3(%24: index, %25: f64):  // 2 preds: ^bb2, ^bb4
    %26 = builtin.unrealized_conversion_cast %24 : index to i64
    %27 = arith.cmpi slt, %24, %23 : index
    cf.cond_br %27, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %28 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.getelementptr %28[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %30 = llvm.load %29 : !llvm.ptr -> i64
    %31 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.getelementptr %31[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %33 = llvm.load %32 : !llvm.ptr -> f64
    %34 = llvm.extractvalue %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.getelementptr %34[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %36 = llvm.load %35 : !llvm.ptr -> f64
    %37 = arith.mulf %33, %36 : f64
    %38 = arith.addf %25, %37 : f64
    %39 = arith.addi %24, %c1 : index
    cf.br ^bb3(%39, %38 : index, f64)
  ^bb5:  // pred: ^bb3
    %40 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.getelementptr %40[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %25, %41 : f64, !llvm.ptr
    %42 = arith.addi %8, %c1 : index
    cf.br ^bb1(%42 : index)
  ^bb6:  // pred: ^bb1
    return %arg2 : memref<3xf64>
  }
  func.func @main(%arg0: memref<3x4xf64>, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<3x4xf64> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c3 = arith.constant 3 : index
    %1 = builtin.unrealized_conversion_cast %c3 : index to i64
    %c4 = arith.constant 4 : index
    %2 = builtin.unrealized_conversion_cast %c4 : index to i64
    %c0 = arith.constant 0 : index
    %3 = builtin.unrealized_conversion_cast %c0 : index to i64
    %c1 = arith.constant 1 : index
    %4 = builtin.unrealized_conversion_cast %c1 : index to i64
    %cst = arith.constant 0.000000e+00 : f64
    %c4_i8 = arith.constant 4 : i8
    %c8_i8 = arith.constant 8 : i8
    %5 = llvm.mlir.constant(2 : index) : i64
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.alloca %5 x i8 : (i64) -> !llvm.ptr
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %7, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.mlir.constant(0 : index) : i64
    %12 = llvm.insertvalue %11, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %5, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.insertvalue %6, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = builtin.unrealized_conversion_cast %14 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xi8>
    %16 = llvm.getelementptr %7[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %c4_i8, %16 : i8, !llvm.ptr
    %17 = llvm.getelementptr %7[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %c8_i8, %17 : i8, !llvm.ptr
    %18 = llvm.mlir.constant(2 : index) : i64
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.alloca %18 x i64 : (i64) -> !llvm.ptr
    %21 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.insertvalue %20, %21[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.insertvalue %20, %22[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.mlir.constant(0 : index) : i64
    %25 = llvm.insertvalue %24, %23[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.insertvalue %18, %25[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.insertvalue %19, %26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = builtin.unrealized_conversion_cast %27 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %29 = llvm.getelementptr %20[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %1, %29 : i64, !llvm.ptr
    %30 = llvm.getelementptr %20[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %2, %30 : i64, !llvm.ptr
    %31 = llvm.mlir.constant(2 : index) : i64
    %32 = llvm.mlir.constant(1 : index) : i64
    %33 = llvm.alloca %31 x i64 : (i64) -> !llvm.ptr
    %34 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %35 = llvm.insertvalue %33, %34[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.insertvalue %33, %35[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.mlir.constant(0 : index) : i64
    %38 = llvm.insertvalue %37, %36[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %39 = llvm.insertvalue %31, %38[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.insertvalue %32, %39[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = builtin.unrealized_conversion_cast %40 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %42 = llvm.getelementptr %33[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %3, %42 : i64, !llvm.ptr
    %43 = llvm.getelementptr %33[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %4, %43 : i64, !llvm.ptr
    %44 = llvm.mlir.zero : !llvm.ptr
    %45 = call @newSparseTensor(%28, %28, %15, %41, %41, %c0_i32, %c0_i32, %c1_i32, %c0_i32, %44) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    %46 = llvm.mlir.constant(2 : index) : i64
    %47 = llvm.mlir.constant(1 : index) : i64
    %48 = llvm.alloca %46 x i64 : (i64) -> !llvm.ptr
    %49 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %50 = llvm.insertvalue %48, %49[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %51 = llvm.insertvalue %48, %50[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %52 = llvm.mlir.constant(0 : index) : i64
    %53 = llvm.insertvalue %52, %51[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %54 = llvm.insertvalue %46, %53[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.insertvalue %47, %54[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = builtin.unrealized_conversion_cast %55 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %57 = llvm.mlir.constant(1 : index) : i64
    %58 = llvm.alloca %57 x f64 : (i64) -> !llvm.ptr
    %59 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %60 = llvm.insertvalue %58, %59[0] : !llvm.struct<(ptr, ptr, i64)> 
    %61 = llvm.insertvalue %58, %60[1] : !llvm.struct<(ptr, ptr, i64)> 
    %62 = llvm.mlir.constant(0 : index) : i64
    %63 = llvm.insertvalue %62, %61[2] : !llvm.struct<(ptr, ptr, i64)> 
    %64 = builtin.unrealized_conversion_cast %63 : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    cf.br ^bb1(%c0 : index)
  ^bb1(%65: index):  // 2 preds: ^bb0, ^bb7
    %66 = builtin.unrealized_conversion_cast %65 : index to i64
    %67 = arith.cmpi slt, %65, %c3 : index
    cf.cond_br %67, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%68: index):  // 2 preds: ^bb2, ^bb6
    %69 = builtin.unrealized_conversion_cast %68 : index to i64
    %70 = arith.cmpi slt, %68, %c4 : index
    cf.cond_br %70, ^bb4, ^bb7
  ^bb4:  // pred: ^bb3
    %71 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.mlir.constant(4 : index) : i64
    %73 = llvm.mul %66, %72  : i64
    %74 = llvm.add %73, %69  : i64
    %75 = llvm.getelementptr %71[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %76 = llvm.load %75 : !llvm.ptr -> f64
    %77 = arith.cmpf une, %76, %cst : f64
    cf.cond_br %77, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %78 = llvm.getelementptr %48[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %66, %78 : i64, !llvm.ptr
    %79 = llvm.getelementptr %48[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %69, %79 : i64, !llvm.ptr
    llvm.store %76, %58 : f64, !llvm.ptr
    call @lexInsertF64(%45, %56, %64) : (!llvm.ptr, memref<?xindex>, memref<f64>) -> ()
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %80 = arith.addi %68, %c1 : index
    cf.br ^bb3(%80 : index)
  ^bb7:  // pred: ^bb3
    %81 = arith.addi %65, %c1 : index
    cf.br ^bb1(%81 : index)
  ^bb8:  // pred: ^bb1
    call @endLexInsert(%45) : (!llvm.ptr) -> ()
    %82 = call @spMV(%45, %arg1, %arg2) : (!llvm.ptr, memref<4xf64>, memref<3xf64>) -> memref<3xf64>
    return %82 : memref<3xf64>
  }
}


// -----// IR Dump After ConvertVectorToLLVMPass (convert-vector-to-llvm) //----- //
module {
  func.func private @endLexInsert(!llvm.ptr)
  func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}
  func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
  func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}
  func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func @spMV(%arg0: !llvm.ptr, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<3xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<4xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %2 = call @sparsePositions0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %3 = builtin.unrealized_conversion_cast %2 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %4 = call @sparseCoordinates0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %5 = builtin.unrealized_conversion_cast %4 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %6 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
    %7 = builtin.unrealized_conversion_cast %6 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    cf.br ^bb1(%c0 : index)
  ^bb1(%8: index):  // 2 preds: ^bb0, ^bb5
    %9 = builtin.unrealized_conversion_cast %8 : index to i64
    %10 = arith.cmpi slt, %8, %c3 : index
    cf.cond_br %10, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %11 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.getelementptr %11[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %13 = llvm.load %12 : !llvm.ptr -> f64
    %14 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.getelementptr %14[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %16 = llvm.load %15 : !llvm.ptr -> i64
    %17 = builtin.unrealized_conversion_cast %16 : i64 to index
    %18 = arith.addi %8, %c1 : index
    %19 = builtin.unrealized_conversion_cast %18 : index to i64
    %20 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.getelementptr %20[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %22 = llvm.load %21 : !llvm.ptr -> i64
    %23 = builtin.unrealized_conversion_cast %22 : i64 to index
    cf.br ^bb3(%17, %13 : index, f64)
  ^bb3(%24: index, %25: f64):  // 2 preds: ^bb2, ^bb4
    %26 = builtin.unrealized_conversion_cast %24 : index to i64
    %27 = arith.cmpi slt, %24, %23 : index
    cf.cond_br %27, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %28 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.getelementptr %28[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %30 = llvm.load %29 : !llvm.ptr -> i64
    %31 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.getelementptr %31[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %33 = llvm.load %32 : !llvm.ptr -> f64
    %34 = llvm.extractvalue %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.getelementptr %34[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %36 = llvm.load %35 : !llvm.ptr -> f64
    %37 = arith.mulf %33, %36 : f64
    %38 = arith.addf %25, %37 : f64
    %39 = arith.addi %24, %c1 : index
    cf.br ^bb3(%39, %38 : index, f64)
  ^bb5:  // pred: ^bb3
    %40 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.getelementptr %40[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %25, %41 : f64, !llvm.ptr
    %42 = arith.addi %8, %c1 : index
    cf.br ^bb1(%42 : index)
  ^bb6:  // pred: ^bb1
    return %arg2 : memref<3xf64>
  }
  func.func @main(%arg0: memref<3x4xf64>, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(4 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(2 : index) : i64
    %c8_i8 = arith.constant 8 : i8
    %c4_i8 = arith.constant 4 : i8
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %4 = builtin.unrealized_conversion_cast %arg0 : memref<3x4xf64> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %5 = builtin.unrealized_conversion_cast %c3 : index to i64
    %6 = builtin.unrealized_conversion_cast %c4 : index to i64
    %7 = builtin.unrealized_conversion_cast %c0 : index to i64
    %8 = builtin.unrealized_conversion_cast %c1 : index to i64
    %9 = llvm.alloca %3 x i8 : (i64) -> !llvm.ptr
    %10 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %9, %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %1, %12[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.insertvalue %3, %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.insertvalue %2, %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = builtin.unrealized_conversion_cast %15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xi8>
    %17 = llvm.getelementptr %9[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %c4_i8, %17 : i8, !llvm.ptr
    %18 = llvm.getelementptr %9[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %c8_i8, %18 : i8, !llvm.ptr
    %19 = llvm.alloca %3 x i64 : (i64) -> !llvm.ptr
    %20 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.insertvalue %19, %20[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.insertvalue %19, %21[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.insertvalue %1, %22[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.insertvalue %3, %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.insertvalue %2, %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = builtin.unrealized_conversion_cast %25 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %27 = llvm.getelementptr %19[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %5, %27 : i64, !llvm.ptr
    %28 = llvm.getelementptr %19[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %6, %28 : i64, !llvm.ptr
    %29 = llvm.alloca %3 x i64 : (i64) -> !llvm.ptr
    %30 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.insertvalue %29, %30[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.insertvalue %29, %31[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.insertvalue %1, %32[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.insertvalue %3, %33[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.insertvalue %2, %34[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = builtin.unrealized_conversion_cast %35 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %37 = llvm.getelementptr %29[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %37 : i64, !llvm.ptr
    %38 = llvm.getelementptr %29[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %8, %38 : i64, !llvm.ptr
    %39 = llvm.mlir.zero : !llvm.ptr
    %40 = call @newSparseTensor(%26, %26, %16, %36, %36, %c0_i32, %c0_i32, %c1_i32, %c0_i32, %39) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    %41 = llvm.alloca %3 x i64 : (i64) -> !llvm.ptr
    %42 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.insertvalue %41, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %44 = llvm.insertvalue %41, %43[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.insertvalue %1, %44[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.insertvalue %3, %45[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.insertvalue %2, %46[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %48 = builtin.unrealized_conversion_cast %47 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %49 = llvm.alloca %2 x f64 : (i64) -> !llvm.ptr
    %50 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %51 = llvm.insertvalue %49, %50[0] : !llvm.struct<(ptr, ptr, i64)> 
    %52 = llvm.insertvalue %49, %51[1] : !llvm.struct<(ptr, ptr, i64)> 
    %53 = llvm.insertvalue %1, %52[2] : !llvm.struct<(ptr, ptr, i64)> 
    %54 = builtin.unrealized_conversion_cast %53 : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    cf.br ^bb1(%c0 : index)
  ^bb1(%55: index):  // 2 preds: ^bb0, ^bb7
    %56 = builtin.unrealized_conversion_cast %55 : index to i64
    %57 = arith.cmpi slt, %55, %c3 : index
    cf.cond_br %57, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%58: index):  // 2 preds: ^bb2, ^bb6
    %59 = builtin.unrealized_conversion_cast %58 : index to i64
    %60 = arith.cmpi slt, %58, %c4 : index
    cf.cond_br %60, ^bb4, ^bb7
  ^bb4:  // pred: ^bb3
    %61 = llvm.extractvalue %4[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.mul %56, %0  : i64
    %63 = llvm.add %62, %59  : i64
    %64 = llvm.getelementptr %61[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %65 = llvm.load %64 : !llvm.ptr -> f64
    %66 = arith.cmpf une, %65, %cst : f64
    cf.cond_br %66, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %67 = llvm.getelementptr %41[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %56, %67 : i64, !llvm.ptr
    %68 = llvm.getelementptr %41[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %59, %68 : i64, !llvm.ptr
    llvm.store %65, %49 : f64, !llvm.ptr
    call @lexInsertF64(%40, %48, %54) : (!llvm.ptr, memref<?xindex>, memref<f64>) -> ()
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %69 = arith.addi %58, %c1 : index
    cf.br ^bb3(%69 : index)
  ^bb7:  // pred: ^bb3
    %70 = arith.addi %55, %c1 : index
    cf.br ^bb1(%70 : index)
  ^bb8:  // pred: ^bb1
    call @endLexInsert(%40) : (!llvm.ptr) -> ()
    %71 = call @spMV(%40, %arg1, %arg2) : (!llvm.ptr, memref<4xf64>, memref<3xf64>) -> memref<3xf64>
    return %71 : memref<3xf64>
  }
}


// -----// IR Dump After ConvertComplexToLLVMPass (convert-complex-to-llvm) //----- //
module {
  func.func private @endLexInsert(!llvm.ptr)
  func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}
  func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
  func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}
  func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func @spMV(%arg0: !llvm.ptr, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<3xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<4xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %2 = call @sparsePositions0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %3 = builtin.unrealized_conversion_cast %2 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %4 = call @sparseCoordinates0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %5 = builtin.unrealized_conversion_cast %4 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %6 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
    %7 = builtin.unrealized_conversion_cast %6 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    cf.br ^bb1(%c0 : index)
  ^bb1(%8: index):  // 2 preds: ^bb0, ^bb5
    %9 = builtin.unrealized_conversion_cast %8 : index to i64
    %10 = arith.cmpi slt, %8, %c3 : index
    cf.cond_br %10, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %11 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.getelementptr %11[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %13 = llvm.load %12 : !llvm.ptr -> f64
    %14 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.getelementptr %14[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %16 = llvm.load %15 : !llvm.ptr -> i64
    %17 = builtin.unrealized_conversion_cast %16 : i64 to index
    %18 = arith.addi %8, %c1 : index
    %19 = builtin.unrealized_conversion_cast %18 : index to i64
    %20 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.getelementptr %20[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %22 = llvm.load %21 : !llvm.ptr -> i64
    %23 = builtin.unrealized_conversion_cast %22 : i64 to index
    cf.br ^bb3(%17, %13 : index, f64)
  ^bb3(%24: index, %25: f64):  // 2 preds: ^bb2, ^bb4
    %26 = builtin.unrealized_conversion_cast %24 : index to i64
    %27 = arith.cmpi slt, %24, %23 : index
    cf.cond_br %27, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %28 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.getelementptr %28[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %30 = llvm.load %29 : !llvm.ptr -> i64
    %31 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.getelementptr %31[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %33 = llvm.load %32 : !llvm.ptr -> f64
    %34 = llvm.extractvalue %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.getelementptr %34[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %36 = llvm.load %35 : !llvm.ptr -> f64
    %37 = arith.mulf %33, %36 : f64
    %38 = arith.addf %25, %37 : f64
    %39 = arith.addi %24, %c1 : index
    cf.br ^bb3(%39, %38 : index, f64)
  ^bb5:  // pred: ^bb3
    %40 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.getelementptr %40[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %25, %41 : f64, !llvm.ptr
    %42 = arith.addi %8, %c1 : index
    cf.br ^bb1(%42 : index)
  ^bb6:  // pred: ^bb1
    return %arg2 : memref<3xf64>
  }
  func.func @main(%arg0: memref<3x4xf64>, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(4 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(2 : index) : i64
    %c8_i8 = arith.constant 8 : i8
    %c4_i8 = arith.constant 4 : i8
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %4 = builtin.unrealized_conversion_cast %arg0 : memref<3x4xf64> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %5 = builtin.unrealized_conversion_cast %c3 : index to i64
    %6 = builtin.unrealized_conversion_cast %c4 : index to i64
    %7 = builtin.unrealized_conversion_cast %c0 : index to i64
    %8 = builtin.unrealized_conversion_cast %c1 : index to i64
    %9 = llvm.alloca %3 x i8 : (i64) -> !llvm.ptr
    %10 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %9, %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %1, %12[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.insertvalue %3, %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.insertvalue %2, %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = builtin.unrealized_conversion_cast %15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xi8>
    %17 = llvm.getelementptr %9[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %c4_i8, %17 : i8, !llvm.ptr
    %18 = llvm.getelementptr %9[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %c8_i8, %18 : i8, !llvm.ptr
    %19 = llvm.alloca %3 x i64 : (i64) -> !llvm.ptr
    %20 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.insertvalue %19, %20[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.insertvalue %19, %21[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.insertvalue %1, %22[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.insertvalue %3, %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.insertvalue %2, %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = builtin.unrealized_conversion_cast %25 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %27 = llvm.getelementptr %19[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %5, %27 : i64, !llvm.ptr
    %28 = llvm.getelementptr %19[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %6, %28 : i64, !llvm.ptr
    %29 = llvm.alloca %3 x i64 : (i64) -> !llvm.ptr
    %30 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.insertvalue %29, %30[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.insertvalue %29, %31[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.insertvalue %1, %32[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.insertvalue %3, %33[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.insertvalue %2, %34[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = builtin.unrealized_conversion_cast %35 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %37 = llvm.getelementptr %29[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %37 : i64, !llvm.ptr
    %38 = llvm.getelementptr %29[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %8, %38 : i64, !llvm.ptr
    %39 = llvm.mlir.zero : !llvm.ptr
    %40 = call @newSparseTensor(%26, %26, %16, %36, %36, %c0_i32, %c0_i32, %c1_i32, %c0_i32, %39) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    %41 = llvm.alloca %3 x i64 : (i64) -> !llvm.ptr
    %42 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.insertvalue %41, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %44 = llvm.insertvalue %41, %43[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.insertvalue %1, %44[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.insertvalue %3, %45[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.insertvalue %2, %46[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %48 = builtin.unrealized_conversion_cast %47 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %49 = llvm.alloca %2 x f64 : (i64) -> !llvm.ptr
    %50 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %51 = llvm.insertvalue %49, %50[0] : !llvm.struct<(ptr, ptr, i64)> 
    %52 = llvm.insertvalue %49, %51[1] : !llvm.struct<(ptr, ptr, i64)> 
    %53 = llvm.insertvalue %1, %52[2] : !llvm.struct<(ptr, ptr, i64)> 
    %54 = builtin.unrealized_conversion_cast %53 : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    cf.br ^bb1(%c0 : index)
  ^bb1(%55: index):  // 2 preds: ^bb0, ^bb7
    %56 = builtin.unrealized_conversion_cast %55 : index to i64
    %57 = arith.cmpi slt, %55, %c3 : index
    cf.cond_br %57, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%58: index):  // 2 preds: ^bb2, ^bb6
    %59 = builtin.unrealized_conversion_cast %58 : index to i64
    %60 = arith.cmpi slt, %58, %c4 : index
    cf.cond_br %60, ^bb4, ^bb7
  ^bb4:  // pred: ^bb3
    %61 = llvm.extractvalue %4[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.mul %56, %0  : i64
    %63 = llvm.add %62, %59  : i64
    %64 = llvm.getelementptr %61[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %65 = llvm.load %64 : !llvm.ptr -> f64
    %66 = arith.cmpf une, %65, %cst : f64
    cf.cond_br %66, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %67 = llvm.getelementptr %41[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %56, %67 : i64, !llvm.ptr
    %68 = llvm.getelementptr %41[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %59, %68 : i64, !llvm.ptr
    llvm.store %65, %49 : f64, !llvm.ptr
    call @lexInsertF64(%40, %48, %54) : (!llvm.ptr, memref<?xindex>, memref<f64>) -> ()
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %69 = arith.addi %58, %c1 : index
    cf.br ^bb3(%69 : index)
  ^bb7:  // pred: ^bb3
    %70 = arith.addi %55, %c1 : index
    cf.br ^bb1(%70 : index)
  ^bb8:  // pred: ^bb1
    call @endLexInsert(%40) : (!llvm.ptr) -> ()
    %71 = call @spMV(%40, %arg1, %arg2) : (!llvm.ptr, memref<4xf64>, memref<3xf64>) -> memref<3xf64>
    return %71 : memref<3xf64>
  }
}


// -----// IR Dump After ConvertVectorToLLVMPass (convert-vector-to-llvm) //----- //
module {
  func.func private @endLexInsert(!llvm.ptr)
  func.func private @lexInsertF64(!llvm.ptr, memref<?xindex>, memref<f64>) attributes {llvm.emit_c_interface}
  func.func private @newSparseTensor(memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface}
  func.func private @sparseValuesF64(!llvm.ptr) -> memref<?xf64> attributes {llvm.emit_c_interface}
  func.func private @sparseCoordinates0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @sparsePositions0(!llvm.ptr, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func @spMV(%arg0: !llvm.ptr, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<3xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<4xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %2 = call @sparsePositions0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %3 = builtin.unrealized_conversion_cast %2 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %4 = call @sparseCoordinates0(%arg0, %c1) : (!llvm.ptr, index) -> memref<?xindex>
    %5 = builtin.unrealized_conversion_cast %4 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %6 = call @sparseValuesF64(%arg0) : (!llvm.ptr) -> memref<?xf64>
    %7 = builtin.unrealized_conversion_cast %6 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    cf.br ^bb1(%c0 : index)
  ^bb1(%8: index):  // 2 preds: ^bb0, ^bb5
    %9 = builtin.unrealized_conversion_cast %8 : index to i64
    %10 = arith.cmpi slt, %8, %c3 : index
    cf.cond_br %10, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %11 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.getelementptr %11[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %13 = llvm.load %12 : !llvm.ptr -> f64
    %14 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.getelementptr %14[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %16 = llvm.load %15 : !llvm.ptr -> i64
    %17 = builtin.unrealized_conversion_cast %16 : i64 to index
    %18 = arith.addi %8, %c1 : index
    %19 = builtin.unrealized_conversion_cast %18 : index to i64
    %20 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.getelementptr %20[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %22 = llvm.load %21 : !llvm.ptr -> i64
    %23 = builtin.unrealized_conversion_cast %22 : i64 to index
    cf.br ^bb3(%17, %13 : index, f64)
  ^bb3(%24: index, %25: f64):  // 2 preds: ^bb2, ^bb4
    %26 = builtin.unrealized_conversion_cast %24 : index to i64
    %27 = arith.cmpi slt, %24, %23 : index
    cf.cond_br %27, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %28 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.getelementptr %28[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %30 = llvm.load %29 : !llvm.ptr -> i64
    %31 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.getelementptr %31[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %33 = llvm.load %32 : !llvm.ptr -> f64
    %34 = llvm.extractvalue %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.getelementptr %34[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %36 = llvm.load %35 : !llvm.ptr -> f64
    %37 = arith.mulf %33, %36 : f64
    %38 = arith.addf %25, %37 : f64
    %39 = arith.addi %24, %c1 : index
    cf.br ^bb3(%39, %38 : index, f64)
  ^bb5:  // pred: ^bb3
    %40 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.getelementptr %40[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %25, %41 : f64, !llvm.ptr
    %42 = arith.addi %8, %c1 : index
    cf.br ^bb1(%42 : index)
  ^bb6:  // pred: ^bb1
    return %arg2 : memref<3xf64>
  }
  func.func @main(%arg0: memref<3x4xf64>, %arg1: memref<4xf64>, %arg2: memref<3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(4 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(2 : index) : i64
    %c8_i8 = arith.constant 8 : i8
    %c4_i8 = arith.constant 4 : i8
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %4 = builtin.unrealized_conversion_cast %arg0 : memref<3x4xf64> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %5 = builtin.unrealized_conversion_cast %c3 : index to i64
    %6 = builtin.unrealized_conversion_cast %c4 : index to i64
    %7 = builtin.unrealized_conversion_cast %c0 : index to i64
    %8 = builtin.unrealized_conversion_cast %c1 : index to i64
    %9 = llvm.alloca %3 x i8 : (i64) -> !llvm.ptr
    %10 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %9, %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %1, %12[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.insertvalue %3, %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.insertvalue %2, %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = builtin.unrealized_conversion_cast %15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xi8>
    %17 = llvm.getelementptr %9[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %c4_i8, %17 : i8, !llvm.ptr
    %18 = llvm.getelementptr %9[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %c8_i8, %18 : i8, !llvm.ptr
    %19 = llvm.alloca %3 x i64 : (i64) -> !llvm.ptr
    %20 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.insertvalue %19, %20[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.insertvalue %19, %21[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.insertvalue %1, %22[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.insertvalue %3, %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.insertvalue %2, %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = builtin.unrealized_conversion_cast %25 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %27 = llvm.getelementptr %19[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %5, %27 : i64, !llvm.ptr
    %28 = llvm.getelementptr %19[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %6, %28 : i64, !llvm.ptr
    %29 = llvm.alloca %3 x i64 : (i64) -> !llvm.ptr
    %30 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.insertvalue %29, %30[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.insertvalue %29, %31[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.insertvalue %1, %32[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.insertvalue %3, %33[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.insertvalue %2, %34[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = builtin.unrealized_conversion_cast %35 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %37 = llvm.getelementptr %29[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %37 : i64, !llvm.ptr
    %38 = llvm.getelementptr %29[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %8, %38 : i64, !llvm.ptr
    %39 = llvm.mlir.zero : !llvm.ptr
    %40 = call @newSparseTensor(%26, %26, %16, %36, %36, %c0_i32, %c0_i32, %c1_i32, %c0_i32, %39) : (memref<?xindex>, memref<?xindex>, memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    %41 = llvm.alloca %3 x i64 : (i64) -> !llvm.ptr
    %42 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.insertvalue %41, %42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %44 = llvm.insertvalue %41, %43[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.insertvalue %1, %44[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.insertvalue %3, %45[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.insertvalue %2, %46[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %48 = builtin.unrealized_conversion_cast %47 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %49 = llvm.alloca %2 x f64 : (i64) -> !llvm.ptr
    %50 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %51 = llvm.insertvalue %49, %50[0] : !llvm.struct<(ptr, ptr, i64)> 
    %52 = llvm.insertvalue %49, %51[1] : !llvm.struct<(ptr, ptr, i64)> 
    %53 = llvm.insertvalue %1, %52[2] : !llvm.struct<(ptr, ptr, i64)> 
    %54 = builtin.unrealized_conversion_cast %53 : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    cf.br ^bb1(%c0 : index)
  ^bb1(%55: index):  // 2 preds: ^bb0, ^bb7
    %56 = builtin.unrealized_conversion_cast %55 : index to i64
    %57 = arith.cmpi slt, %55, %c3 : index
    cf.cond_br %57, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%58: index):  // 2 preds: ^bb2, ^bb6
    %59 = builtin.unrealized_conversion_cast %58 : index to i64
    %60 = arith.cmpi slt, %58, %c4 : index
    cf.cond_br %60, ^bb4, ^bb7
  ^bb4:  // pred: ^bb3
    %61 = llvm.extractvalue %4[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.mul %56, %0  : i64
    %63 = llvm.add %62, %59  : i64
    %64 = llvm.getelementptr %61[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %65 = llvm.load %64 : !llvm.ptr -> f64
    %66 = arith.cmpf une, %65, %cst : f64
    cf.cond_br %66, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %67 = llvm.getelementptr %41[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %56, %67 : i64, !llvm.ptr
    %68 = llvm.getelementptr %41[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %59, %68 : i64, !llvm.ptr
    llvm.store %65, %49 : f64, !llvm.ptr
    call @lexInsertF64(%40, %48, %54) : (!llvm.ptr, memref<?xindex>, memref<f64>) -> ()
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %69 = arith.addi %58, %c1 : index
    cf.br ^bb3(%69 : index)
  ^bb7:  // pred: ^bb3
    %70 = arith.addi %55, %c1 : index
    cf.br ^bb1(%70 : index)
  ^bb8:  // pred: ^bb1
    call @endLexInsert(%40) : (!llvm.ptr) -> ()
    %71 = call @spMV(%40, %arg1, %arg2) : (!llvm.ptr, memref<4xf64>, memref<3xf64>) -> memref<3xf64>
    return %71 : memref<3xf64>
  }
}


// -----// IR Dump After ConvertFuncToLLVMPass (convert-func-to-llvm) //----- //
module {
  llvm.func @endLexInsert(!llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func private @lexInsertF64(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: i64) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %5, %7 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %9 = llvm.insertvalue %arg6, %8[0] : !llvm.struct<(ptr, ptr, i64)> 
    %10 = llvm.insertvalue %arg7, %9[1] : !llvm.struct<(ptr, ptr, i64)> 
    %11 = llvm.insertvalue %arg8, %10[2] : !llvm.struct<(ptr, ptr, i64)> 
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.alloca %12 x !llvm.struct<(ptr, ptr, i64)> : (i64) -> !llvm.ptr
    llvm.store %11, %13 : !llvm.struct<(ptr, ptr, i64)>, !llvm.ptr
    llvm.call @_mlir_ciface_lexInsertF64(%arg0, %7, %13) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_lexInsertF64(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @newSparseTensor(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr, %arg16: !llvm.ptr, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %5, %7 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg5, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg6, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg7, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg8, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg9, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.alloca %14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %13, %15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %16 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.insertvalue %arg10, %16[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.insertvalue %arg11, %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.insertvalue %arg12, %18[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.insertvalue %arg13, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.insertvalue %arg14, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.mlir.constant(1 : index) : i64
    %23 = llvm.alloca %22 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %21, %23 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %24 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.insertvalue %arg15, %24[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.insertvalue %arg16, %25[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.insertvalue %arg17, %26[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.insertvalue %arg18, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.insertvalue %arg19, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.alloca %30 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %29, %31 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %32 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %33 = llvm.insertvalue %arg20, %32[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.insertvalue %arg21, %33[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.insertvalue %arg22, %34[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.insertvalue %arg23, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.insertvalue %arg24, %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %38 = llvm.mlir.constant(1 : index) : i64
    %39 = llvm.alloca %38 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %37, %39 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %40 = llvm.call @_mlir_ciface_newSparseTensor(%7, %15, %23, %31, %39, %arg25, %arg26, %arg27, %arg28, %arg29) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    llvm.return %40 : !llvm.ptr
  }
  llvm.func @_mlir_ciface_newSparseTensor(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @sparseValuesF64(%arg0: !llvm.ptr) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.call @_mlir_ciface_sparseValuesF64(%1, %arg0) : (!llvm.ptr, !llvm.ptr) -> ()
    %2 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %2 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @_mlir_ciface_sparseValuesF64(!llvm.ptr, !llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @sparseCoordinates0(%arg0: !llvm.ptr, %arg1: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.call @_mlir_ciface_sparseCoordinates0(%1, %arg0, %arg1) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %2 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %2 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @_mlir_ciface_sparseCoordinates0(!llvm.ptr, !llvm.ptr, i64) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @sparsePositions0(%arg0: !llvm.ptr, %arg1: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.call @_mlir_ciface_sparsePositions0(%1, %arg0, %arg1) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %2 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %2 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @_mlir_ciface_sparsePositions0(!llvm.ptr, !llvm.ptr, i64) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @spMV(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: i64, %arg9: i64, %arg10: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = builtin.unrealized_conversion_cast %5 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<4xf64>
    %7 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.insertvalue %arg6, %7[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %arg7, %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg8, %9[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg9, %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg10, %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = builtin.unrealized_conversion_cast %12 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<3xf64>
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = llvm.mlir.constant(3 : index) : i64
    %17 = builtin.unrealized_conversion_cast %13 : memref<3xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %18 = builtin.unrealized_conversion_cast %6 : memref<4xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.call @sparsePositions0(%arg0, %14) : (!llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %20 = builtin.unrealized_conversion_cast %19 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %21 = builtin.unrealized_conversion_cast %20 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.call @sparseCoordinates0(%arg0, %14) : (!llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %23 = builtin.unrealized_conversion_cast %22 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %24 = builtin.unrealized_conversion_cast %23 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.call @sparseValuesF64(%arg0) : (!llvm.ptr) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %26 = builtin.unrealized_conversion_cast %25 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %27 = builtin.unrealized_conversion_cast %26 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.br ^bb1(%15 : i64)
  ^bb1(%28: i64):  // 2 preds: ^bb0, ^bb5
    %29 = builtin.unrealized_conversion_cast %28 : i64 to index
    %30 = builtin.unrealized_conversion_cast %29 : index to i64
    %31 = llvm.icmp "slt" %28, %16 : i64
    llvm.cond_br %31, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %32 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.getelementptr %32[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %34 = llvm.load %33 : !llvm.ptr -> f64
    %35 = llvm.extractvalue %21[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.getelementptr %35[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %37 = llvm.load %36 : !llvm.ptr -> i64
    %38 = builtin.unrealized_conversion_cast %37 : i64 to index
    %39 = llvm.add %28, %14  : i64
    %40 = builtin.unrealized_conversion_cast %39 : i64 to index
    %41 = builtin.unrealized_conversion_cast %40 : index to i64
    %42 = llvm.extractvalue %21[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %43 = llvm.getelementptr %42[%41] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %44 = llvm.load %43 : !llvm.ptr -> i64
    %45 = builtin.unrealized_conversion_cast %44 : i64 to index
    llvm.br ^bb3(%37, %34 : i64, f64)
  ^bb3(%46: i64, %47: f64):  // 2 preds: ^bb2, ^bb4
    %48 = builtin.unrealized_conversion_cast %46 : i64 to index
    %49 = builtin.unrealized_conversion_cast %48 : index to i64
    %50 = llvm.icmp "slt" %46, %44 : i64
    llvm.cond_br %50, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %51 = llvm.extractvalue %24[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %52 = llvm.getelementptr %51[%49] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %53 = llvm.load %52 : !llvm.ptr -> i64
    %54 = llvm.extractvalue %27[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.getelementptr %54[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %56 = llvm.load %55 : !llvm.ptr -> f64
    %57 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.getelementptr %57[%53] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %59 = llvm.load %58 : !llvm.ptr -> f64
    %60 = llvm.fmul %56, %59  : f64
    %61 = llvm.fadd %47, %60  : f64
    %62 = llvm.add %46, %14  : i64
    llvm.br ^bb3(%62, %61 : i64, f64)
  ^bb5:  // pred: ^bb3
    %63 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %64 = llvm.getelementptr %63[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %47, %64 : f64, !llvm.ptr
    %65 = llvm.add %28, %14  : i64
    llvm.br ^bb1(%65 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return %12 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: i64, %arg16: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = builtin.unrealized_conversion_cast %7 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<3x4xf64>
    %9 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %arg7, %9[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg8, %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg9, %11[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg10, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.insertvalue %arg11, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.insertvalue %arg12, %15[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.insertvalue %arg13, %16[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.insertvalue %arg14, %17[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.insertvalue %arg15, %18[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.insertvalue %arg16, %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.mlir.constant(4 : index) : i64
    %22 = llvm.mlir.constant(0 : index) : i64
    %23 = llvm.mlir.constant(1 : index) : i64
    %24 = llvm.mlir.constant(2 : index) : i64
    %25 = llvm.mlir.constant(8 : i8) : i8
    %26 = llvm.mlir.constant(4 : i8) : i8
    %27 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %28 = llvm.mlir.constant(1 : index) : i64
    %29 = builtin.unrealized_conversion_cast %28 : i64 to index
    %30 = llvm.mlir.constant(0 : index) : i64
    %31 = builtin.unrealized_conversion_cast %30 : i64 to index
    %32 = llvm.mlir.constant(4 : index) : i64
    %33 = builtin.unrealized_conversion_cast %32 : i64 to index
    %34 = llvm.mlir.constant(3 : index) : i64
    %35 = builtin.unrealized_conversion_cast %34 : i64 to index
    %36 = llvm.mlir.constant(0 : i32) : i32
    %37 = llvm.mlir.constant(1 : i32) : i32
    %38 = builtin.unrealized_conversion_cast %8 : memref<3x4xf64> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %39 = builtin.unrealized_conversion_cast %35 : index to i64
    %40 = builtin.unrealized_conversion_cast %33 : index to i64
    %41 = builtin.unrealized_conversion_cast %31 : index to i64
    %42 = builtin.unrealized_conversion_cast %29 : index to i64
    %43 = llvm.alloca %24 x i8 : (i64) -> !llvm.ptr
    %44 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %45 = llvm.insertvalue %43, %44[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.insertvalue %43, %45[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.insertvalue %22, %46[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %48 = llvm.insertvalue %24, %47[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %49 = llvm.insertvalue %23, %48[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %50 = builtin.unrealized_conversion_cast %49 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xi8>
    %51 = llvm.getelementptr %43[%41] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %26, %51 : i8, !llvm.ptr
    %52 = llvm.getelementptr %43[%42] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %25, %52 : i8, !llvm.ptr
    %53 = llvm.alloca %24 x i64 : (i64) -> !llvm.ptr
    %54 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %55 = llvm.insertvalue %53, %54[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = llvm.insertvalue %53, %55[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = llvm.insertvalue %22, %56[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.insertvalue %24, %57[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %59 = llvm.insertvalue %23, %58[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %60 = builtin.unrealized_conversion_cast %59 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %61 = llvm.getelementptr %53[%41] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %39, %61 : i64, !llvm.ptr
    %62 = llvm.getelementptr %53[%42] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %40, %62 : i64, !llvm.ptr
    %63 = llvm.alloca %24 x i64 : (i64) -> !llvm.ptr
    %64 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %65 = llvm.insertvalue %63, %64[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %66 = llvm.insertvalue %63, %65[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %67 = llvm.insertvalue %22, %66[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %68 = llvm.insertvalue %24, %67[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %69 = llvm.insertvalue %23, %68[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %70 = builtin.unrealized_conversion_cast %69 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %71 = llvm.getelementptr %63[%41] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %41, %71 : i64, !llvm.ptr
    %72 = llvm.getelementptr %63[%42] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %42, %72 : i64, !llvm.ptr
    %73 = llvm.mlir.zero : !llvm.ptr
    %74 = llvm.extractvalue %59[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %75 = llvm.extractvalue %59[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %76 = llvm.extractvalue %59[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %77 = llvm.extractvalue %59[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %78 = llvm.extractvalue %59[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %79 = llvm.extractvalue %59[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %80 = llvm.extractvalue %59[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %81 = llvm.extractvalue %59[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %82 = llvm.extractvalue %59[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %83 = llvm.extractvalue %59[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %84 = llvm.extractvalue %49[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %85 = llvm.extractvalue %49[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %86 = llvm.extractvalue %49[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %87 = llvm.extractvalue %49[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %88 = llvm.extractvalue %49[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %89 = llvm.extractvalue %69[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %90 = llvm.extractvalue %69[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %91 = llvm.extractvalue %69[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %92 = llvm.extractvalue %69[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %93 = llvm.extractvalue %69[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %94 = llvm.extractvalue %69[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %95 = llvm.extractvalue %69[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %96 = llvm.extractvalue %69[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %97 = llvm.extractvalue %69[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %98 = llvm.extractvalue %69[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %99 = llvm.call @newSparseTensor(%74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %36, %36, %37, %36, %73) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    %100 = llvm.alloca %24 x i64 : (i64) -> !llvm.ptr
    %101 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %102 = llvm.insertvalue %100, %101[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %103 = llvm.insertvalue %100, %102[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %104 = llvm.insertvalue %22, %103[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %105 = llvm.insertvalue %24, %104[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %106 = llvm.insertvalue %23, %105[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %107 = builtin.unrealized_conversion_cast %106 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %108 = llvm.alloca %23 x f64 : (i64) -> !llvm.ptr
    %109 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %110 = llvm.insertvalue %108, %109[0] : !llvm.struct<(ptr, ptr, i64)> 
    %111 = llvm.insertvalue %108, %110[1] : !llvm.struct<(ptr, ptr, i64)> 
    %112 = llvm.insertvalue %22, %111[2] : !llvm.struct<(ptr, ptr, i64)> 
    %113 = builtin.unrealized_conversion_cast %112 : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    llvm.br ^bb1(%30 : i64)
  ^bb1(%114: i64):  // 2 preds: ^bb0, ^bb7
    %115 = builtin.unrealized_conversion_cast %114 : i64 to index
    %116 = builtin.unrealized_conversion_cast %115 : index to i64
    %117 = llvm.icmp "slt" %114, %34 : i64
    llvm.cond_br %117, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%30 : i64)
  ^bb3(%118: i64):  // 2 preds: ^bb2, ^bb6
    %119 = builtin.unrealized_conversion_cast %118 : i64 to index
    %120 = builtin.unrealized_conversion_cast %119 : index to i64
    %121 = llvm.icmp "slt" %118, %32 : i64
    llvm.cond_br %121, ^bb4, ^bb7
  ^bb4:  // pred: ^bb3
    %122 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %123 = llvm.mul %116, %21  : i64
    %124 = llvm.add %123, %120  : i64
    %125 = llvm.getelementptr %122[%124] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %126 = llvm.load %125 : !llvm.ptr -> f64
    %127 = llvm.fcmp "une" %126, %27 : f64
    llvm.cond_br %127, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %128 = llvm.getelementptr %100[%41] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %116, %128 : i64, !llvm.ptr
    %129 = llvm.getelementptr %100[%42] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %120, %129 : i64, !llvm.ptr
    llvm.store %126, %108 : f64, !llvm.ptr
    %130 = llvm.extractvalue %106[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %131 = llvm.extractvalue %106[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %132 = llvm.extractvalue %106[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %133 = llvm.extractvalue %106[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %134 = llvm.extractvalue %106[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %135 = llvm.extractvalue %112[0] : !llvm.struct<(ptr, ptr, i64)> 
    %136 = llvm.extractvalue %112[1] : !llvm.struct<(ptr, ptr, i64)> 
    %137 = llvm.extractvalue %112[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.call @lexInsertF64(%99, %130, %131, %132, %133, %134, %135, %136, %137) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %138 = llvm.add %118, %28  : i64
    llvm.br ^bb3(%138 : i64)
  ^bb7:  // pred: ^bb3
    %139 = llvm.add %114, %28  : i64
    llvm.br ^bb1(%139 : i64)
  ^bb8:  // pred: ^bb1
    llvm.call @endLexInsert(%99) : (!llvm.ptr) -> ()
    %140 = llvm.extractvalue %14[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %141 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %142 = llvm.extractvalue %14[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %143 = llvm.extractvalue %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %144 = llvm.extractvalue %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %145 = llvm.extractvalue %20[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %146 = llvm.extractvalue %20[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %147 = llvm.extractvalue %20[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %148 = llvm.extractvalue %20[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %149 = llvm.extractvalue %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %150 = llvm.call @spMV(%99, %140, %141, %142, %143, %144, %145, %146, %147, %148, %149) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %150 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @_mlir_ciface_main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.load %arg3 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.extractvalue %14[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.extractvalue %14[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.extractvalue %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.extractvalue %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.call @main(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %15, %16, %17, %18, %19) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.store %20, %arg0 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.return
  }
}


// -----// IR Dump After ReconcileUnrealizedCasts (reconcile-unrealized-casts) //----- //
module {
  llvm.func @endLexInsert(!llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func private @lexInsertF64(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: i64) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %5, %7 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %9 = llvm.insertvalue %arg6, %8[0] : !llvm.struct<(ptr, ptr, i64)> 
    %10 = llvm.insertvalue %arg7, %9[1] : !llvm.struct<(ptr, ptr, i64)> 
    %11 = llvm.insertvalue %arg8, %10[2] : !llvm.struct<(ptr, ptr, i64)> 
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.alloca %12 x !llvm.struct<(ptr, ptr, i64)> : (i64) -> !llvm.ptr
    llvm.store %11, %13 : !llvm.struct<(ptr, ptr, i64)>, !llvm.ptr
    llvm.call @_mlir_ciface_lexInsertF64(%arg0, %7, %13) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_lexInsertF64(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @newSparseTensor(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr, %arg16: !llvm.ptr, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %5, %7 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg5, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg6, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg7, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg8, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg9, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.alloca %14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %13, %15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %16 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.insertvalue %arg10, %16[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.insertvalue %arg11, %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.insertvalue %arg12, %18[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.insertvalue %arg13, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.insertvalue %arg14, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.mlir.constant(1 : index) : i64
    %23 = llvm.alloca %22 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %21, %23 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %24 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.insertvalue %arg15, %24[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.insertvalue %arg16, %25[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.insertvalue %arg17, %26[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.insertvalue %arg18, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.insertvalue %arg19, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.alloca %30 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %29, %31 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %32 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %33 = llvm.insertvalue %arg20, %32[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.insertvalue %arg21, %33[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.insertvalue %arg22, %34[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.insertvalue %arg23, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.insertvalue %arg24, %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %38 = llvm.mlir.constant(1 : index) : i64
    %39 = llvm.alloca %38 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %37, %39 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %40 = llvm.call @_mlir_ciface_newSparseTensor(%7, %15, %23, %31, %39, %arg25, %arg26, %arg27, %arg28, %arg29) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    llvm.return %40 : !llvm.ptr
  }
  llvm.func @_mlir_ciface_newSparseTensor(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @sparseValuesF64(%arg0: !llvm.ptr) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.call @_mlir_ciface_sparseValuesF64(%1, %arg0) : (!llvm.ptr, !llvm.ptr) -> ()
    %2 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %2 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @_mlir_ciface_sparseValuesF64(!llvm.ptr, !llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @sparseCoordinates0(%arg0: !llvm.ptr, %arg1: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.call @_mlir_ciface_sparseCoordinates0(%1, %arg0, %arg1) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %2 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %2 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @_mlir_ciface_sparseCoordinates0(!llvm.ptr, !llvm.ptr, i64) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @sparsePositions0(%arg0: !llvm.ptr, %arg1: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.call @_mlir_ciface_sparsePositions0(%1, %arg0, %arg1) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %2 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %2 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @_mlir_ciface_sparsePositions0(!llvm.ptr, !llvm.ptr, i64) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @spMV(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: i64, %arg9: i64, %arg10: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.insertvalue %arg7, %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %arg8, %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg9, %9[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg10, %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.mlir.constant(3 : index) : i64
    %15 = llvm.call @sparsePositions0(%arg0, %12) : (!llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.call @sparseCoordinates0(%arg0, %12) : (!llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.call @sparseValuesF64(%arg0) : (!llvm.ptr) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.br ^bb1(%13 : i64)
  ^bb1(%18: i64):  // 2 preds: ^bb0, ^bb5
    %19 = llvm.icmp "slt" %18, %14 : i64
    llvm.cond_br %19, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %20 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.getelementptr %20[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %22 = llvm.load %21 : !llvm.ptr -> f64
    %23 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.getelementptr %23[%18] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %25 = llvm.load %24 : !llvm.ptr -> i64
    %26 = llvm.add %18, %12  : i64
    %27 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.getelementptr %27[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %29 = llvm.load %28 : !llvm.ptr -> i64
    llvm.br ^bb3(%25, %22 : i64, f64)
  ^bb3(%30: i64, %31: f64):  // 2 preds: ^bb2, ^bb4
    %32 = llvm.icmp "slt" %30, %29 : i64
    llvm.cond_br %32, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %33 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.getelementptr %33[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %35 = llvm.load %34 : !llvm.ptr -> i64
    %36 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.getelementptr %36[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %38 = llvm.load %37 : !llvm.ptr -> f64
    %39 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.getelementptr %39[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %41 = llvm.load %40 : !llvm.ptr -> f64
    %42 = llvm.fmul %38, %41  : f64
    %43 = llvm.fadd %31, %42  : f64
    %44 = llvm.add %30, %12  : i64
    llvm.br ^bb3(%44, %43 : i64, f64)
  ^bb5:  // pred: ^bb3
    %45 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.getelementptr %45[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %31, %46 : f64, !llvm.ptr
    %47 = llvm.add %18, %12  : i64
    llvm.br ^bb1(%47 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return %11 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: i64, %arg16: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg11, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.insertvalue %arg12, %14[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.insertvalue %arg13, %15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.insertvalue %arg14, %16[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.insertvalue %arg15, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.insertvalue %arg16, %18[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.mlir.constant(4 : index) : i64
    %21 = llvm.mlir.constant(0 : index) : i64
    %22 = llvm.mlir.constant(1 : index) : i64
    %23 = llvm.mlir.constant(2 : index) : i64
    %24 = llvm.mlir.constant(8 : i8) : i8
    %25 = llvm.mlir.constant(4 : i8) : i8
    %26 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %27 = llvm.mlir.constant(1 : index) : i64
    %28 = llvm.mlir.constant(0 : index) : i64
    %29 = llvm.mlir.constant(4 : index) : i64
    %30 = llvm.mlir.constant(3 : index) : i64
    %31 = llvm.mlir.constant(0 : i32) : i32
    %32 = llvm.mlir.constant(1 : i32) : i32
    %33 = llvm.alloca %23 x i8 : (i64) -> !llvm.ptr
    %34 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %35 = llvm.insertvalue %33, %34[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.insertvalue %33, %35[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.insertvalue %21, %36[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %38 = llvm.insertvalue %23, %37[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %39 = llvm.insertvalue %22, %38[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.getelementptr %33[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %25, %40 : i8, !llvm.ptr
    %41 = llvm.getelementptr %33[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %24, %41 : i8, !llvm.ptr
    %42 = llvm.alloca %23 x i64 : (i64) -> !llvm.ptr
    %43 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %44 = llvm.insertvalue %42, %43[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.insertvalue %42, %44[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.insertvalue %21, %45[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.insertvalue %23, %46[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %48 = llvm.insertvalue %22, %47[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %49 = llvm.getelementptr %42[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %30, %49 : i64, !llvm.ptr
    %50 = llvm.getelementptr %42[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %29, %50 : i64, !llvm.ptr
    %51 = llvm.alloca %23 x i64 : (i64) -> !llvm.ptr
    %52 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %53 = llvm.insertvalue %51, %52[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %54 = llvm.insertvalue %51, %53[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.insertvalue %21, %54[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = llvm.insertvalue %23, %55[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = llvm.insertvalue %22, %56[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.getelementptr %51[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %28, %58 : i64, !llvm.ptr
    %59 = llvm.getelementptr %51[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %27, %59 : i64, !llvm.ptr
    %60 = llvm.mlir.zero : !llvm.ptr
    %61 = llvm.call @newSparseTensor(%42, %42, %21, %23, %22, %42, %42, %21, %23, %22, %33, %33, %21, %23, %22, %51, %51, %21, %23, %22, %51, %51, %21, %23, %22, %31, %31, %32, %31, %60) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    %62 = llvm.alloca %23 x i64 : (i64) -> !llvm.ptr
    %63 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %64 = llvm.insertvalue %62, %63[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %65 = llvm.insertvalue %62, %64[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %66 = llvm.insertvalue %21, %65[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %67 = llvm.insertvalue %23, %66[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %68 = llvm.insertvalue %22, %67[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %69 = llvm.alloca %22 x f64 : (i64) -> !llvm.ptr
    %70 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %71 = llvm.insertvalue %69, %70[0] : !llvm.struct<(ptr, ptr, i64)> 
    %72 = llvm.insertvalue %69, %71[1] : !llvm.struct<(ptr, ptr, i64)> 
    %73 = llvm.insertvalue %21, %72[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.br ^bb1(%28 : i64)
  ^bb1(%74: i64):  // 2 preds: ^bb0, ^bb7
    %75 = llvm.icmp "slt" %74, %30 : i64
    llvm.cond_br %75, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%28 : i64)
  ^bb3(%76: i64):  // 2 preds: ^bb2, ^bb6
    %77 = llvm.icmp "slt" %76, %29 : i64
    llvm.cond_br %77, ^bb4, ^bb7
  ^bb4:  // pred: ^bb3
    %78 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.mul %74, %20  : i64
    %80 = llvm.add %79, %76  : i64
    %81 = llvm.getelementptr %78[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %82 = llvm.load %81 : !llvm.ptr -> f64
    %83 = llvm.fcmp "une" %82, %26 : f64
    llvm.cond_br %83, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %84 = llvm.getelementptr %62[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %74, %84 : i64, !llvm.ptr
    %85 = llvm.getelementptr %62[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %76, %85 : i64, !llvm.ptr
    llvm.store %82, %69 : f64, !llvm.ptr
    llvm.call @lexInsertF64(%61, %62, %62, %21, %23, %22, %69, %69, %21) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %86 = llvm.add %76, %27  : i64
    llvm.br ^bb3(%86 : i64)
  ^bb7:  // pred: ^bb3
    %87 = llvm.add %74, %27  : i64
    llvm.br ^bb1(%87 : i64)
  ^bb8:  // pred: ^bb1
    llvm.call @endLexInsert(%61) : (!llvm.ptr) -> ()
    %88 = llvm.call @spMV(%61, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %88 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @_mlir_ciface_main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.load %arg3 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.extractvalue %14[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.extractvalue %14[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.extractvalue %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.extractvalue %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.call @main(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %15, %16, %17, %18, %19) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.store %20, %arg0 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.return
  }
}


module {
  llvm.func @endLexInsert(!llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func private @lexInsertF64(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: i64) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %5, %7 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %9 = llvm.insertvalue %arg6, %8[0] : !llvm.struct<(ptr, ptr, i64)> 
    %10 = llvm.insertvalue %arg7, %9[1] : !llvm.struct<(ptr, ptr, i64)> 
    %11 = llvm.insertvalue %arg8, %10[2] : !llvm.struct<(ptr, ptr, i64)> 
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.alloca %12 x !llvm.struct<(ptr, ptr, i64)> : (i64) -> !llvm.ptr
    llvm.store %11, %13 : !llvm.struct<(ptr, ptr, i64)>, !llvm.ptr
    llvm.call @_mlir_ciface_lexInsertF64(%arg0, %7, %13) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_lexInsertF64(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @newSparseTensor(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr, %arg16: !llvm.ptr, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr, %arg21: !llvm.ptr, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %5, %7 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg5, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg6, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg7, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg8, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg9, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.alloca %14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %13, %15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %16 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.insertvalue %arg10, %16[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.insertvalue %arg11, %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.insertvalue %arg12, %18[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.insertvalue %arg13, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.insertvalue %arg14, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.mlir.constant(1 : index) : i64
    %23 = llvm.alloca %22 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %21, %23 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %24 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.insertvalue %arg15, %24[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.insertvalue %arg16, %25[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.insertvalue %arg17, %26[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.insertvalue %arg18, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.insertvalue %arg19, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.alloca %30 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %29, %31 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %32 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %33 = llvm.insertvalue %arg20, %32[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.insertvalue %arg21, %33[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.insertvalue %arg22, %34[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.insertvalue %arg23, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.insertvalue %arg24, %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %38 = llvm.mlir.constant(1 : index) : i64
    %39 = llvm.alloca %38 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %37, %39 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %40 = llvm.call @_mlir_ciface_newSparseTensor(%7, %15, %23, %31, %39, %arg25, %arg26, %arg27, %arg28, %arg29) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    llvm.return %40 : !llvm.ptr
  }
  llvm.func @_mlir_ciface_newSparseTensor(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @sparseValuesF64(%arg0: !llvm.ptr) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.call @_mlir_ciface_sparseValuesF64(%1, %arg0) : (!llvm.ptr, !llvm.ptr) -> ()
    %2 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %2 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @_mlir_ciface_sparseValuesF64(!llvm.ptr, !llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @sparseCoordinates0(%arg0: !llvm.ptr, %arg1: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.call @_mlir_ciface_sparseCoordinates0(%1, %arg0, %arg1) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %2 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %2 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @_mlir_ciface_sparseCoordinates0(!llvm.ptr, !llvm.ptr, i64) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @sparsePositions0(%arg0: !llvm.ptr, %arg1: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.call @_mlir_ciface_sparsePositions0(%1, %arg0, %arg1) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %2 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %2 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @_mlir_ciface_sparsePositions0(!llvm.ptr, !llvm.ptr, i64) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @spMV(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: i64, %arg9: i64, %arg10: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.insertvalue %arg7, %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %arg8, %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg9, %9[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg10, %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.mlir.constant(3 : index) : i64
    %15 = llvm.call @sparsePositions0(%arg0, %12) : (!llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.call @sparseCoordinates0(%arg0, %12) : (!llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.call @sparseValuesF64(%arg0) : (!llvm.ptr) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.br ^bb1(%13 : i64)
  ^bb1(%18: i64):  // 2 preds: ^bb0, ^bb5
    %19 = llvm.icmp "slt" %18, %14 : i64
    llvm.cond_br %19, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %20 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.getelementptr %20[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %22 = llvm.load %21 : !llvm.ptr -> f64
    %23 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.getelementptr %23[%18] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %25 = llvm.load %24 : !llvm.ptr -> i64
    %26 = llvm.add %18, %12  : i64
    %27 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.getelementptr %27[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %29 = llvm.load %28 : !llvm.ptr -> i64
    llvm.br ^bb3(%25, %22 : i64, f64)
  ^bb3(%30: i64, %31: f64):  // 2 preds: ^bb2, ^bb4
    %32 = llvm.icmp "slt" %30, %29 : i64
    llvm.cond_br %32, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %33 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.getelementptr %33[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %35 = llvm.load %34 : !llvm.ptr -> i64
    %36 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.getelementptr %36[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %38 = llvm.load %37 : !llvm.ptr -> f64
    %39 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.getelementptr %39[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %41 = llvm.load %40 : !llvm.ptr -> f64
    %42 = llvm.fmul %38, %41  : f64
    %43 = llvm.fadd %31, %42  : f64
    %44 = llvm.add %30, %12  : i64
    llvm.br ^bb3(%44, %43 : i64, f64)
  ^bb5:  // pred: ^bb3
    %45 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.getelementptr %45[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %31, %46 : f64, !llvm.ptr
    %47 = llvm.add %18, %12  : i64
    llvm.br ^bb1(%47 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return %11 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: i64, %arg16: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg11, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.insertvalue %arg12, %14[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.insertvalue %arg13, %15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.insertvalue %arg14, %16[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.insertvalue %arg15, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.insertvalue %arg16, %18[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.mlir.constant(4 : index) : i64
    %21 = llvm.mlir.constant(0 : index) : i64
    %22 = llvm.mlir.constant(1 : index) : i64
    %23 = llvm.mlir.constant(2 : index) : i64
    %24 = llvm.mlir.constant(8 : i8) : i8
    %25 = llvm.mlir.constant(4 : i8) : i8
    %26 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %27 = llvm.mlir.constant(1 : index) : i64
    %28 = llvm.mlir.constant(0 : index) : i64
    %29 = llvm.mlir.constant(4 : index) : i64
    %30 = llvm.mlir.constant(3 : index) : i64
    %31 = llvm.mlir.constant(0 : i32) : i32
    %32 = llvm.mlir.constant(1 : i32) : i32
    %33 = llvm.alloca %23 x i8 : (i64) -> !llvm.ptr
    %34 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %35 = llvm.insertvalue %33, %34[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.insertvalue %33, %35[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.insertvalue %21, %36[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %38 = llvm.insertvalue %23, %37[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %39 = llvm.insertvalue %22, %38[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.getelementptr %33[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %25, %40 : i8, !llvm.ptr
    %41 = llvm.getelementptr %33[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %24, %41 : i8, !llvm.ptr
    %42 = llvm.alloca %23 x i64 : (i64) -> !llvm.ptr
    %43 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %44 = llvm.insertvalue %42, %43[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.insertvalue %42, %44[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.insertvalue %21, %45[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.insertvalue %23, %46[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %48 = llvm.insertvalue %22, %47[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %49 = llvm.getelementptr %42[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %30, %49 : i64, !llvm.ptr
    %50 = llvm.getelementptr %42[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %29, %50 : i64, !llvm.ptr
    %51 = llvm.alloca %23 x i64 : (i64) -> !llvm.ptr
    %52 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %53 = llvm.insertvalue %51, %52[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %54 = llvm.insertvalue %51, %53[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.insertvalue %21, %54[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = llvm.insertvalue %23, %55[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = llvm.insertvalue %22, %56[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.getelementptr %51[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %28, %58 : i64, !llvm.ptr
    %59 = llvm.getelementptr %51[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %27, %59 : i64, !llvm.ptr
    %60 = llvm.mlir.zero : !llvm.ptr
    %61 = llvm.call @newSparseTensor(%42, %42, %21, %23, %22, %42, %42, %21, %23, %22, %33, %33, %21, %23, %22, %51, %51, %21, %23, %22, %51, %51, %21, %23, %22, %31, %31, %32, %31, %60) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i32, i32, i32, i32, !llvm.ptr) -> !llvm.ptr
    %62 = llvm.alloca %23 x i64 : (i64) -> !llvm.ptr
    %63 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %64 = llvm.insertvalue %62, %63[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %65 = llvm.insertvalue %62, %64[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %66 = llvm.insertvalue %21, %65[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %67 = llvm.insertvalue %23, %66[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %68 = llvm.insertvalue %22, %67[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %69 = llvm.alloca %22 x f64 : (i64) -> !llvm.ptr
    %70 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %71 = llvm.insertvalue %69, %70[0] : !llvm.struct<(ptr, ptr, i64)> 
    %72 = llvm.insertvalue %69, %71[1] : !llvm.struct<(ptr, ptr, i64)> 
    %73 = llvm.insertvalue %21, %72[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.br ^bb1(%28 : i64)
  ^bb1(%74: i64):  // 2 preds: ^bb0, ^bb7
    %75 = llvm.icmp "slt" %74, %30 : i64
    llvm.cond_br %75, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%28 : i64)
  ^bb3(%76: i64):  // 2 preds: ^bb2, ^bb6
    %77 = llvm.icmp "slt" %76, %29 : i64
    llvm.cond_br %77, ^bb4, ^bb7
  ^bb4:  // pred: ^bb3
    %78 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.mul %74, %20  : i64
    %80 = llvm.add %79, %76  : i64
    %81 = llvm.getelementptr %78[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %82 = llvm.load %81 : !llvm.ptr -> f64
    %83 = llvm.fcmp "une" %82, %26 : f64
    llvm.cond_br %83, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %84 = llvm.getelementptr %62[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %74, %84 : i64, !llvm.ptr
    %85 = llvm.getelementptr %62[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %76, %85 : i64, !llvm.ptr
    llvm.store %82, %69 : f64, !llvm.ptr
    llvm.call @lexInsertF64(%61, %62, %62, %21, %23, %22, %69, %69, %21) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %86 = llvm.add %76, %27  : i64
    llvm.br ^bb3(%86 : i64)
  ^bb7:  // pred: ^bb3
    %87 = llvm.add %74, %27  : i64
    llvm.br ^bb1(%87 : i64)
  ^bb8:  // pred: ^bb1
    llvm.call @endLexInsert(%61) : (!llvm.ptr) -> ()
    %88 = llvm.call @spMV(%61, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return %88 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
  llvm.func @_mlir_ciface_main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.load %arg3 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.extractvalue %14[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.extractvalue %14[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.extractvalue %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.extractvalue %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.call @main(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %15, %16, %17, %18, %19) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.store %20, %arg0 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.return
  }
}

