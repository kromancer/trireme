; ModuleID = 'sddmm.c'
source_filename = "sddmm.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.19.0"

; Function Attrs: noinline nounwind ssp
define void @compute(double* noalias %A_vals, i32* noalias %B1_pos, i32* noalias %B1_crd, i32* noalias %B2_pos, i32* noalias %B2_crd, double* noalias %B_vals, i32 %C2_dimension, double* noalias %C_vals, i32 %D1_dimension, i32 %D2_dimension, double* noalias %D_vals) #0 {
entry:
  %A_vals.addr = alloca double*, align 4
  %B1_pos.addr = alloca i32*, align 4
  %B1_crd.addr = alloca i32*, align 4
  %B2_pos.addr = alloca i32*, align 4
  %B2_crd.addr = alloca i32*, align 4
  %B_vals.addr = alloca double*, align 4
  %C2_dimension.addr = alloca i32, align 4
  %C_vals.addr = alloca double*, align 4
  %D1_dimension.addr = alloca i32, align 4
  %D2_dimension.addr = alloca i32, align 4
  %D_vals.addr = alloca double*, align 4
  %jA = alloca i32, align 4
  %iB = alloca i32, align 4
  %i = alloca i32, align 4
  %k = alloca i32, align 4
  %kC = alloca i32, align 4
  %jB = alloca i32, align 4
  %j = alloca i32, align 4
  %jD = alloca i32, align 4
  store double* %A_vals, double** %A_vals.addr, align 4
  store i32* %B1_pos, i32** %B1_pos.addr, align 4
  store i32* %B1_crd, i32** %B1_crd.addr, align 4
  store i32* %B2_pos, i32** %B2_pos.addr, align 4
  store i32* %B2_crd, i32** %B2_crd.addr, align 4
  store double* %B_vals, double** %B_vals.addr, align 4
  store i32 %C2_dimension, i32* %C2_dimension.addr, align 4
  store double* %C_vals, double** %C_vals.addr, align 4
  store i32 %D1_dimension, i32* %D1_dimension.addr, align 4
  store i32 %D2_dimension, i32* %D2_dimension.addr, align 4
  store double* %D_vals, double** %D_vals.addr, align 4
  store i32 0, i32* %jA, align 4
  %0 = load i32*, i32** %B1_pos.addr, align 4
  %arrayidx = getelementptr inbounds i32, i32* %0, i32 0
  %1 = load i32, i32* %arrayidx, align 4
  store i32 %1, i32* %iB, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc28, %entry
  %2 = load i32, i32* %iB, align 4
  %3 = load i32*, i32** %B1_pos.addr, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %3, i32 1
  %4 = load i32, i32* %arrayidx1, align 4
  %cmp = icmp slt i32 %2, %4
  br i1 %cmp, label %for.body, label %for.end30

for.body:                                         ; preds = %for.cond
  %5 = load i32*, i32** %B1_crd.addr, align 4
  %6 = load i32, i32* %iB, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %5, i32 %6
  %7 = load i32, i32* %arrayidx2, align 4
  store i32 %7, i32* %i, align 4
  store i32 0, i32* %k, align 4
  br label %for.cond3

for.cond3:                                        ; preds = %for.inc25, %for.body
  %8 = load i32, i32* %k, align 4
  %9 = load i32, i32* %D1_dimension.addr, align 4
  %cmp4 = icmp slt i32 %8, %9
  br i1 %cmp4, label %for.body5, label %for.end27

for.body5:                                        ; preds = %for.cond3
  %10 = load i32, i32* %i, align 4
  %11 = load i32, i32* %C2_dimension.addr, align 4
  %mul = mul nsw i32 %10, %11
  %12 = load i32, i32* %k, align 4
  %add = add nsw i32 %mul, %12
  store i32 %add, i32* %kC, align 4
  %13 = load i32*, i32** %B2_pos.addr, align 4
  %14 = load i32, i32* %iB, align 4
  %arrayidx6 = getelementptr inbounds i32, i32* %13, i32 %14
  %15 = load i32, i32* %arrayidx6, align 4
  store i32 %15, i32* %jB, align 4
  br label %for.cond7

for.cond7:                                        ; preds = %for.inc, %for.body5
  %16 = load i32, i32* %jB, align 4
  %17 = load i32*, i32** %B2_pos.addr, align 4
  %18 = load i32, i32* %iB, align 4
  %add8 = add nsw i32 %18, 1
  %arrayidx9 = getelementptr inbounds i32, i32* %17, i32 %add8
  %19 = load i32, i32* %arrayidx9, align 4
  %cmp10 = icmp slt i32 %16, %19
  br i1 %cmp10, label %for.body11, label %for.end

for.body11:                                       ; preds = %for.cond7
  %20 = load i32*, i32** %B2_crd.addr, align 4
  %21 = load i32, i32* %jB, align 4
  %arrayidx12 = getelementptr inbounds i32, i32* %20, i32 %21
  %22 = load i32, i32* %arrayidx12, align 4
  store i32 %22, i32* %j, align 4
  %23 = load double*, double** %A_vals.addr, align 4
  %24 = load i32, i32* %jA, align 4
  %arrayidx13 = getelementptr inbounds double, double* %23, i32 %24
  store double 0.000000e+00, double* %arrayidx13, align 4
  %25 = load i32, i32* %k, align 4
  %26 = load i32, i32* %D2_dimension.addr, align 4
  %mul14 = mul nsw i32 %25, %26
  %27 = load i32, i32* %j, align 4
  %add15 = add nsw i32 %mul14, %27
  store i32 %add15, i32* %jD, align 4
  %28 = load double*, double** %A_vals.addr, align 4
  %29 = load i32, i32* %jA, align 4
  %arrayidx16 = getelementptr inbounds double, double* %28, i32 %29
  %30 = load double, double* %arrayidx16, align 4
  %31 = load double*, double** %B_vals.addr, align 4
  %32 = load i32, i32* %jB, align 4
  %arrayidx17 = getelementptr inbounds double, double* %31, i32 %32
  %33 = load double, double* %arrayidx17, align 4
  %34 = load double*, double** %C_vals.addr, align 4
  %35 = load i32, i32* %kC, align 4
  %arrayidx18 = getelementptr inbounds double, double* %34, i32 %35
  %36 = load double, double* %arrayidx18, align 4
  %mul19 = fmul double %33, %36
  %37 = load double*, double** %D_vals.addr, align 4
  %38 = load i32, i32* %jD, align 4
  %arrayidx20 = getelementptr inbounds double, double* %37, i32 %38
  %39 = load double, double* %arrayidx20, align 4
  %mul21 = fmul double %mul19, %39
  %add22 = fadd double %30, %mul21
  %40 = load double*, double** %A_vals.addr, align 4
  %41 = load i32, i32* %jA, align 4
  %arrayidx23 = getelementptr inbounds double, double* %40, i32 %41
  store double %add22, double* %arrayidx23, align 4
  %42 = load i32, i32* %jA, align 4
  %inc = add nsw i32 %42, 1
  store i32 %inc, i32* %jA, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body11
  %43 = load i32, i32* %jB, align 4
  %inc24 = add nsw i32 %43, 1
  store i32 %inc24, i32* %jB, align 4
  br label %for.cond7

for.end:                                          ; preds = %for.cond7
  br label %for.inc25

for.inc25:                                        ; preds = %for.end
  %44 = load i32, i32* %k, align 4
  %inc26 = add nsw i32 %44, 1
  store i32 %inc26, i32* %k, align 4
  br label %for.cond3

for.end27:                                        ; preds = %for.cond3
  br label %for.inc28

for.inc28:                                        ; preds = %for.end27
  %45 = load i32, i32* %iB, align 4
  %inc29 = add nsw i32 %45, 1
  store i32 %inc29, i32* %iB, align 4
  br label %for.cond

for.end30:                                        ; preds = %for.cond
  ret void
}

attributes #0 = { noinline nounwind ssp "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+armv4t,+soft-float,+strict-align,-crypto,-d32,-dotprod,-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fp16,-fp16fml,-fp64,-fpregs,-fullfp16,-mve,-mve.fp,-neon,-thumb-mode,-vfp2,-vfp2sp,-vfp3,-vfp3d16,-vfp3d16sp,-vfp3sp,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp" "unsafe-fp-math"="false" "use-soft-float"="true" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{!"clang version 10.0.0 (git@github.com:kromancer/llvm-project.git d32170dbd5b0d54436537b6b75beaf44324e0c28)"}
