; ModuleID = 'spmv-coo-dense.c'
source_filename = "spmv-coo-dense.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.19.0"

; Function Attrs: noinline nounwind ssp
define i32 @compute(double* noalias %a_vals, i32* noalias %B1_pos, i32* noalias %B1_crd, i32* noalias %B2_crd, double* noalias %B_vals, double* noalias %c_vals) #0 {
entry:
  %a_vals.addr = alloca double*, align 4
  %B1_pos.addr = alloca i32*, align 4
  %B1_crd.addr = alloca i32*, align 4
  %B2_crd.addr = alloca i32*, align 4
  %B_vals.addr = alloca double*, align 4
  %c_vals.addr = alloca double*, align 4
  %iB = alloca i32, align 4
  %pB1_end = alloca i32, align 4
  %i = alloca i32, align 4
  %B1_segend = alloca i32, align 4
  %tja_val = alloca double, align 8
  %jB = alloca i32, align 4
  %j = alloca i32, align 4
  store double* %a_vals, double** %a_vals.addr, align 4
  store i32* %B1_pos, i32** %B1_pos.addr, align 4
  store i32* %B1_crd, i32** %B1_crd.addr, align 4
  store i32* %B2_crd, i32** %B2_crd.addr, align 4
  store double* %B_vals, double** %B_vals.addr, align 4
  store double* %c_vals, double** %c_vals.addr, align 4
  %0 = load i32*, i32** %B1_pos.addr, align 4
  %arrayidx = getelementptr inbounds i32, i32* %0, i32 0
  %1 = load i32, i32* %arrayidx, align 4
  store i32 %1, i32* %iB, align 4
  %2 = load i32*, i32** %B1_pos.addr, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %2, i32 1
  %3 = load i32, i32* %arrayidx1, align 4
  store i32 %3, i32* %pB1_end, align 4
  br label %while.cond

while.cond:                                       ; preds = %for.end, %entry
  %4 = load i32, i32* %iB, align 4
  %5 = load i32, i32* %pB1_end, align 4
  %cmp = icmp slt i32 %4, %5
  br i1 %cmp, label %while.body, label %while.end15

while.body:                                       ; preds = %while.cond
  %6 = load i32*, i32** %B1_crd.addr, align 4
  %7 = load i32, i32* %iB, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %6, i32 %7
  %8 = load i32, i32* %arrayidx2, align 4
  store i32 %8, i32* %i, align 4
  %9 = load i32, i32* %iB, align 4
  %add = add nsw i32 %9, 1
  store i32 %add, i32* %B1_segend, align 4
  br label %while.cond3

while.cond3:                                      ; preds = %while.body7, %while.body
  %10 = load i32, i32* %B1_segend, align 4
  %11 = load i32, i32* %pB1_end, align 4
  %cmp4 = icmp slt i32 %10, %11
  br i1 %cmp4, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %while.cond3
  %12 = load i32*, i32** %B1_crd.addr, align 4
  %13 = load i32, i32* %B1_segend, align 4
  %arrayidx5 = getelementptr inbounds i32, i32* %12, i32 %13
  %14 = load i32, i32* %arrayidx5, align 4
  %15 = load i32, i32* %i, align 4
  %cmp6 = icmp eq i32 %14, %15
  br label %land.end

land.end:                                         ; preds = %land.rhs, %while.cond3
  %16 = phi i1 [ false, %while.cond3 ], [ %cmp6, %land.rhs ]
  br i1 %16, label %while.body7, label %while.end

while.body7:                                      ; preds = %land.end
  %17 = load i32, i32* %B1_segend, align 4
  %inc = add nsw i32 %17, 1
  store i32 %inc, i32* %B1_segend, align 4
  br label %while.cond3

while.end:                                        ; preds = %land.end
  store double 0.000000e+00, double* %tja_val, align 8
  %18 = load i32, i32* %iB, align 4
  store i32 %18, i32* %jB, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %while.end
  %19 = load i32, i32* %jB, align 4
  %20 = load i32, i32* %B1_segend, align 4
  %cmp8 = icmp slt i32 %19, %20
  br i1 %cmp8, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %21 = load i32*, i32** %B2_crd.addr, align 4
  %22 = load i32, i32* %jB, align 4
  %arrayidx9 = getelementptr inbounds i32, i32* %21, i32 %22
  %23 = load i32, i32* %arrayidx9, align 4
  store i32 %23, i32* %j, align 4
  %24 = load double*, double** %B_vals.addr, align 4
  %25 = load i32, i32* %jB, align 4
  %arrayidx10 = getelementptr inbounds double, double* %24, i32 %25
  %26 = load double, double* %arrayidx10, align 4
  %27 = load double*, double** %c_vals.addr, align 4
  %28 = load i32, i32* %j, align 4
  %arrayidx11 = getelementptr inbounds double, double* %27, i32 %28
  %29 = load double, double* %arrayidx11, align 4
  %mul = fmul double %26, %29
  %30 = load double, double* %tja_val, align 8
  %add12 = fadd double %30, %mul
  store double %add12, double* %tja_val, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %31 = load i32, i32* %jB, align 4
  %inc13 = add nsw i32 %31, 1
  store i32 %inc13, i32* %jB, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %32 = load double, double* %tja_val, align 8
  %33 = load double*, double** %a_vals.addr, align 4
  %34 = load i32, i32* %i, align 4
  %arrayidx14 = getelementptr inbounds double, double* %33, i32 %34
  store double %32, double* %arrayidx14, align 4
  %35 = load i32, i32* %B1_segend, align 4
  store i32 %35, i32* %iB, align 4
  br label %while.cond

while.end15:                                      ; preds = %while.cond
  ret i32 0
}

attributes #0 = { noinline nounwind ssp "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+armv4t,+soft-float,+strict-align,-crypto,-d32,-dotprod,-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fp16,-fp16fml,-fp64,-fpregs,-fullfp16,-mve,-mve.fp,-neon,-thumb-mode,-vfp2,-vfp2sp,-vfp3,-vfp3d16,-vfp3d16sp,-vfp3sp,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp" "unsafe-fp-math"="false" "use-soft-float"="true" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{!"clang version 10.0.0 (git@github.com:kromancer/llvm-project.git d32170dbd5b0d54436537b6b75beaf44324e0c28)"}
