; ModuleID = 'spmm-csr-dense.c'
source_filename = "spmm-csr-dense.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.20.0"

; Function Attrs: noinline nounwind ssp
define i32 @compute(i32 %A2_dimension, i32 %B1_dimension, i32 %C2_dimension, double* noalias %A_vals, double* noalias %C_vals, i32* noalias %B2_pos, i32* noalias %B2_crd, double* noalias %B_vals) #0 {
entry:
  %A2_dimension.addr = alloca i32, align 4
  %B1_dimension.addr = alloca i32, align 4
  %C2_dimension.addr = alloca i32, align 4
  %A_vals.addr = alloca double*, align 4
  %C_vals.addr = alloca double*, align 4
  %B2_pos.addr = alloca i32*, align 4
  %B2_crd.addr = alloca i32*, align 4
  %B_vals.addr = alloca double*, align 4
  %i = alloca i32, align 4
  %jB = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %kA = alloca i32, align 4
  %kC = alloca i32, align 4
  store i32 %A2_dimension, i32* %A2_dimension.addr, align 4
  store i32 %B1_dimension, i32* %B1_dimension.addr, align 4
  store i32 %C2_dimension, i32* %C2_dimension.addr, align 4
  store double* %A_vals, double** %A_vals.addr, align 4
  store double* %C_vals, double** %C_vals.addr, align 4
  store i32* %B2_pos, i32** %B2_pos.addr, align 4
  store i32* %B2_crd, i32** %B2_crd.addr, align 4
  store double* %B_vals, double** %B_vals.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc21, %entry
  %0 = load i32, i32* %i, align 4
  %1 = load i32, i32* %B1_dimension.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end23

for.body:                                         ; preds = %for.cond
  %2 = load i32*, i32** %B2_pos.addr, align 4
  %3 = load i32, i32* %i, align 4
  %arrayidx = getelementptr inbounds i32, i32* %2, i32 %3
  %4 = load i32, i32* %arrayidx, align 4
  store i32 %4, i32* %jB, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc18, %for.body
  %5 = load i32, i32* %jB, align 4
  %6 = load i32*, i32** %B2_pos.addr, align 4
  %7 = load i32, i32* %i, align 4
  %add = add nsw i32 %7, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %6, i32 %add
  %8 = load i32, i32* %arrayidx2, align 4
  %cmp3 = icmp slt i32 %5, %8
  br i1 %cmp3, label %for.body4, label %for.end20

for.body4:                                        ; preds = %for.cond1
  %9 = load i32*, i32** %B2_crd.addr, align 4
  %10 = load i32, i32* %jB, align 4
  %arrayidx5 = getelementptr inbounds i32, i32* %9, i32 %10
  %11 = load i32, i32* %arrayidx5, align 4
  store i32 %11, i32* %j, align 4
  store i32 0, i32* %k, align 4
  br label %for.cond6

for.cond6:                                        ; preds = %for.inc, %for.body4
  %12 = load i32, i32* %k, align 4
  %13 = load i32, i32* %C2_dimension.addr, align 4
  %cmp7 = icmp slt i32 %12, %13
  br i1 %cmp7, label %for.body8, label %for.end

for.body8:                                        ; preds = %for.cond6
  %14 = load i32, i32* %i, align 4
  %15 = load i32, i32* %A2_dimension.addr, align 4
  %mul = mul nsw i32 %14, %15
  %16 = load i32, i32* %k, align 4
  %add9 = add nsw i32 %mul, %16
  store i32 %add9, i32* %kA, align 4
  %17 = load i32, i32* %j, align 4
  %18 = load i32, i32* %C2_dimension.addr, align 4
  %mul10 = mul nsw i32 %17, %18
  %19 = load i32, i32* %k, align 4
  %add11 = add nsw i32 %mul10, %19
  store i32 %add11, i32* %kC, align 4
  %20 = load double*, double** %A_vals.addr, align 4
  %21 = load i32, i32* %kA, align 4
  %arrayidx12 = getelementptr inbounds double, double* %20, i32 %21
  %22 = load double, double* %arrayidx12, align 4
  %23 = load double*, double** %B_vals.addr, align 4
  %24 = load i32, i32* %jB, align 4
  %arrayidx13 = getelementptr inbounds double, double* %23, i32 %24
  %25 = load double, double* %arrayidx13, align 4
  %26 = load double*, double** %C_vals.addr, align 4
  %27 = load i32, i32* %kC, align 4
  %arrayidx14 = getelementptr inbounds double, double* %26, i32 %27
  %28 = load double, double* %arrayidx14, align 4
  %mul15 = fmul double %25, %28
  %add16 = fadd double %22, %mul15
  %29 = load double*, double** %A_vals.addr, align 4
  %30 = load i32, i32* %kA, align 4
  %arrayidx17 = getelementptr inbounds double, double* %29, i32 %30
  store double %add16, double* %arrayidx17, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body8
  %31 = load i32, i32* %k, align 4
  %inc = add nsw i32 %31, 1
  store i32 %inc, i32* %k, align 4
  br label %for.cond6

for.end:                                          ; preds = %for.cond6
  br label %for.inc18

for.inc18:                                        ; preds = %for.end
  %32 = load i32, i32* %jB, align 4
  %inc19 = add nsw i32 %32, 1
  store i32 %inc19, i32* %jB, align 4
  br label %for.cond1

for.end20:                                        ; preds = %for.cond1
  br label %for.inc21

for.inc21:                                        ; preds = %for.end20
  %33 = load i32, i32* %i, align 4
  %inc22 = add nsw i32 %33, 1
  store i32 %inc22, i32* %i, align 4
  br label %for.cond

for.end23:                                        ; preds = %for.cond
  ret i32 0
}

attributes #0 = { noinline nounwind ssp "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+armv4t,+soft-float,+strict-align,-crypto,-d32,-dotprod,-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fp16,-fp16fml,-fp64,-fpregs,-fullfp16,-mve,-mve.fp,-neon,-thumb-mode,-vfp2,-vfp2sp,-vfp3,-vfp3d16,-vfp3d16sp,-vfp3sp,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp" "unsafe-fp-math"="false" "use-soft-float"="true" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{!"clang version 10.0.0 (git@github.com:kromancer/llvm-project.git d32170dbd5b0d54436537b6b75beaf44324e0c28)"}
