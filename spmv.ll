; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @endLexInsert(ptr)

define private void @lexInsertF64(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5, ptr %6, ptr %7, i64 %8) {
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %1, 0
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, ptr %2, 1
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, i64 %3, 2
  %13 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, i64 %4, 3, 0
  %14 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, i64 %5, 4, 0
  %15 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, ptr %15, align 8
  %16 = insertvalue { ptr, ptr, i64 } undef, ptr %6, 0
  %17 = insertvalue { ptr, ptr, i64 } %16, ptr %7, 1
  %18 = insertvalue { ptr, ptr, i64 } %17, i64 %8, 2
  %19 = alloca { ptr, ptr, i64 }, i64 1, align 8
  store { ptr, ptr, i64 } %18, ptr %19, align 8
  call void @_mlir_ciface_lexInsertF64(ptr %0, ptr %15, ptr %19)
  ret void
}

declare void @_mlir_ciface_lexInsertF64(ptr, ptr, ptr)

define private ptr @newSparseTensor(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, i32 %25, i32 %26, i32 %27, i32 %28, ptr %29) {
  %31 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %0, 0
  %32 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %31, ptr %1, 1
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, i64 %2, 2
  %34 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, i64 %3, 3, 0
  %35 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, i64 %4, 4, 0
  %36 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %35, ptr %36, align 8
  %37 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %5, 0
  %38 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, ptr %6, 1
  %39 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, i64 %7, 2
  %40 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, i64 %8, 3, 0
  %41 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, i64 %9, 4, 0
  %42 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, ptr %42, align 8
  %43 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %10, 0
  %44 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %43, ptr %11, 1
  %45 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, i64 %12, 2
  %46 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %45, i64 %13, 3, 0
  %47 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, i64 %14, 4, 0
  %48 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %47, ptr %48, align 8
  %49 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %15, 0
  %50 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %49, ptr %16, 1
  %51 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, i64 %17, 2
  %52 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, i64 %18, 3, 0
  %53 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, i64 %19, 4, 0
  %54 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %53, ptr %54, align 8
  %55 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %20, 0
  %56 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, ptr %21, 1
  %57 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, i64 %22, 2
  %58 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %57, i64 %23, 3, 0
  %59 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %58, i64 %24, 4, 0
  %60 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %59, ptr %60, align 8
  %61 = call ptr @_mlir_ciface_newSparseTensor(ptr %36, ptr %42, ptr %48, ptr %54, ptr %60, i32 %25, i32 %26, i32 %27, i32 %28, ptr %29)
  ret ptr %61
}

declare ptr @_mlir_ciface_newSparseTensor(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, ptr)

define private { ptr, ptr, i64, [1 x i64], [1 x i64] } @sparseValuesF64(ptr %0) {
  %2 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  call void @_mlir_ciface_sparseValuesF64(ptr %2, ptr %0)
  %3 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %3
}

declare void @_mlir_ciface_sparseValuesF64(ptr, ptr)

define private { ptr, ptr, i64, [1 x i64], [1 x i64] } @sparseCoordinates0(ptr %0, i64 %1) {
  %3 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  call void @_mlir_ciface_sparseCoordinates0(ptr %3, ptr %0, i64 %1)
  %4 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %3, align 8
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %4
}

declare void @_mlir_ciface_sparseCoordinates0(ptr, ptr, i64)

define private { ptr, ptr, i64, [1 x i64], [1 x i64] } @sparsePositions0(ptr %0, i64 %1) {
  %3 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  call void @_mlir_ciface_sparsePositions0(ptr %3, ptr %0, i64 %1)
  %4 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %3, align 8
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %4
}

declare void @_mlir_ciface_sparsePositions0(ptr, ptr, i64)

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @spMV(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5, ptr %6, ptr %7, i64 %8, i64 %9, i64 %10) {
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %1, 0
  %13 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, ptr %2, 1
  %14 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, i64 %3, 2
  %15 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, i64 %4, 3, 0
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, i64 %5, 4, 0
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %6, 0
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, ptr %7, 1
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 %8, 2
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 %9, 3, 0
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, i64 %10, 4, 0
  %22 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @sparsePositions0(ptr %0, i64 1)
  %23 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @sparseCoordinates0(ptr %0, i64 1)
  %24 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @sparseValuesF64(ptr %0)
  br label %25

25:                                               ; preds = %56, %11
  %26 = phi i64 [ %59, %56 ], [ 0, %11 ]
  %27 = icmp slt i64 %26, 3
  br i1 %27, label %28, label %60

28:                                               ; preds = %25
  %29 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, 1
  %30 = getelementptr double, ptr %29, i64 %26
  %31 = load double, ptr %30, align 8
  %32 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, 1
  %33 = getelementptr i64, ptr %32, i64 %26
  %34 = load i64, ptr %33, align 4
  %35 = add i64 %26, 1
  %36 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, 1
  %37 = getelementptr i64, ptr %36, i64 %35
  %38 = load i64, ptr %37, align 4
  br label %39

39:                                               ; preds = %43, %28
  %40 = phi i64 [ %55, %43 ], [ %34, %28 ]
  %41 = phi double [ %54, %43 ], [ %31, %28 ]
  %42 = icmp slt i64 %40, %38
  br i1 %42, label %43, label %56

43:                                               ; preds = %39
  %44 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 1
  %45 = getelementptr i64, ptr %44, i64 %40
  %46 = load i64, ptr %45, align 4
  %47 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, 1
  %48 = getelementptr double, ptr %47, i64 %40
  %49 = load double, ptr %48, align 8
  %50 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 1
  %51 = getelementptr double, ptr %50, i64 %46
  %52 = load double, ptr %51, align 8
  %53 = fmul double %49, %52
  %54 = fadd double %41, %53
  %55 = add i64 %40, 1
  br label %39

56:                                               ; preds = %39
  %57 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, 1
  %58 = getelementptr double, ptr %57, i64 %26
  store double %41, ptr %58, align 8
  %59 = add i64 %26, 1
  br label %25

60:                                               ; preds = %25
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %21
}

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16) {
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %0, 0
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, ptr %1, 1
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 %2, 2
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 %3, 3, 0
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, i64 %5, 4, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, i64 %4, 3, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %6, 4, 1
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %7, 0
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, ptr %8, 1
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, i64 %9, 2
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 %10, 3, 0
  %29 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, i64 %11, 4, 0
  %30 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %12, 0
  %31 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, ptr %13, 1
  %32 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %31, i64 %14, 2
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, i64 %15, 3, 0
  %34 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, i64 %16, 4, 0
  %35 = alloca i8, i64 2, align 1
  %36 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %35, 0
  %37 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, ptr %35, 1
  %38 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, i64 0, 2
  %39 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, i64 2, 3, 0
  %40 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, i64 1, 4, 0
  %41 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, 1
  %42 = getelementptr i8, ptr %41, i64 0
  store i8 4, ptr %42, align 1
  %43 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, 1
  %44 = getelementptr i8, ptr %43, i64 1
  store i8 8, ptr %44, align 1
  %45 = alloca i64, i64 2, align 8
  %46 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %45, 0
  %47 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, ptr %45, 1
  %48 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %47, i64 0, 2
  %49 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %48, i64 2, 3, 0
  %50 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %49, i64 1, 4, 0
  %51 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, 1
  %52 = getelementptr i64, ptr %51, i64 0
  store i64 3, ptr %52, align 4
  %53 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, 1
  %54 = getelementptr i64, ptr %53, i64 1
  store i64 4, ptr %54, align 4
  %55 = alloca i64, i64 2, align 8
  %56 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %55, 0
  %57 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, ptr %55, 1
  %58 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %57, i64 0, 2
  %59 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %58, i64 2, 3, 0
  %60 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %59, i64 1, 4, 0
  %61 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, 1
  %62 = getelementptr i64, ptr %61, i64 0
  store i64 0, ptr %62, align 4
  %63 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, 1
  %64 = getelementptr i64, ptr %63, i64 1
  store i64 1, ptr %64, align 4
  %65 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, 0
  %66 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, 1
  %67 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, 2
  %68 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, 3, 0
  %69 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, 4, 0
  %70 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, 0
  %71 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, 1
  %72 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, 2
  %73 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, 3, 0
  %74 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, 4, 0
  %75 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, 0
  %76 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, 1
  %77 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, 2
  %78 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, 3, 0
  %79 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, 4, 0
  %80 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, 0
  %81 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, 1
  %82 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, 2
  %83 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, 3, 0
  %84 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, 4, 0
  %85 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, 0
  %86 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, 1
  %87 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, 2
  %88 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, 3, 0
  %89 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, 4, 0
  %90 = call ptr @newSparseTensor(ptr %65, ptr %66, i64 %67, i64 %68, i64 %69, ptr %70, ptr %71, i64 %72, i64 %73, i64 %74, ptr %75, ptr %76, i64 %77, i64 %78, i64 %79, ptr %80, ptr %81, i64 %82, i64 %83, i64 %84, ptr %85, ptr %86, i64 %87, i64 %88, i64 %89, i32 0, i32 0, i32 1, i32 0, ptr null)
  %91 = alloca i64, i64 2, align 8
  %92 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %91, 0
  %93 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %92, ptr %91, 1
  %94 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %93, i64 0, 2
  %95 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %94, i64 2, 3, 0
  %96 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %95, i64 1, 4, 0
  %97 = alloca double, i64 1, align 8
  %98 = insertvalue { ptr, ptr, i64 } undef, ptr %97, 0
  %99 = insertvalue { ptr, ptr, i64 } %98, ptr %97, 1
  %100 = insertvalue { ptr, ptr, i64 } %99, i64 0, 2
  br label %101

101:                                              ; preds = %131, %17
  %102 = phi i64 [ %132, %131 ], [ 0, %17 ]
  %103 = icmp slt i64 %102, 3
  br i1 %103, label %104, label %133

104:                                              ; preds = %101
  br label %105

105:                                              ; preds = %129, %104
  %106 = phi i64 [ %130, %129 ], [ 0, %104 ]
  %107 = icmp slt i64 %106, 4
  br i1 %107, label %108, label %131

108:                                              ; preds = %105
  %109 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 1
  %110 = mul i64 %102, 4
  %111 = add i64 %110, %106
  %112 = getelementptr double, ptr %109, i64 %111
  %113 = load double, ptr %112, align 8
  %114 = fcmp une double %113, 0.000000e+00
  br i1 %114, label %115, label %129

115:                                              ; preds = %108
  %116 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %96, 1
  %117 = getelementptr i64, ptr %116, i64 0
  store i64 %102, ptr %117, align 4
  %118 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %96, 1
  %119 = getelementptr i64, ptr %118, i64 1
  store i64 %106, ptr %119, align 4
  %120 = extractvalue { ptr, ptr, i64 } %100, 1
  store double %113, ptr %120, align 8
  %121 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %96, 0
  %122 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %96, 1
  %123 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %96, 2
  %124 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %96, 3, 0
  %125 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %96, 4, 0
  %126 = extractvalue { ptr, ptr, i64 } %100, 0
  %127 = extractvalue { ptr, ptr, i64 } %100, 1
  %128 = extractvalue { ptr, ptr, i64 } %100, 2
  call void @lexInsertF64(ptr %90, ptr %121, ptr %122, i64 %123, i64 %124, i64 %125, ptr %126, ptr %127, i64 %128)
  br label %129

129:                                              ; preds = %115, %108
  %130 = add i64 %106, 1
  br label %105

131:                                              ; preds = %105
  %132 = add i64 %102, 1
  br label %101

133:                                              ; preds = %101
  call void @endLexInsert(ptr %90)
  %134 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, 0
  %135 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, 1
  %136 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, 2
  %137 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, 3, 0
  %138 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, 4, 0
  %139 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, 0
  %140 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, 1
  %141 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, 2
  %142 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, 3, 0
  %143 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, 4, 0
  %144 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @spMV(ptr %90, ptr %134, ptr %135, i64 %136, i64 %137, i64 %138, ptr %139, ptr %140, i64 %141, i64 %142, i64 %143)
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %144
}

define void @_mlir_ciface_main(ptr %0, ptr %1, ptr %2, ptr %3) {
  %5 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %6 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 0
  %7 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 1
  %8 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 2
  %9 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 3, 0
  %10 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 3, 1
  %11 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 4, 0
  %12 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 4, 1
  %13 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 0
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 1
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 2
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 3, 0
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, 4, 0
  %19 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %3, align 8
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 0
  %21 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 1
  %22 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 2
  %23 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 3, 0
  %24 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, 4, 0
  %25 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @main(ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24)
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, ptr %0, align 8
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
