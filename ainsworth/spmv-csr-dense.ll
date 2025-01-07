; ModuleID = 'spmv-csr-dense.c'
source_filename = "spmv-csr-dense.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.20.0"

; Function Attrs: noinline nounwind ssp
define i32 @compute(double* noalias %a_vals, i32 %num_of_rows, i32* noalias %pos, i32* noalias %crd, double* noalias %B_vals, double* noalias %c_vals) #0 {
entry:
  %a_vals.addr = alloca double*, align 4
  %num_of_rows.addr = alloca i32, align 4
  %pos.addr = alloca i32*, align 4
  %crd.addr = alloca i32*, align 4
  %B_vals.addr = alloca double*, align 4
  %c_vals.addr = alloca double*, align 4
  %i = alloca i32, align 4
  %res = alloca double, align 8
  %jj = alloca i32, align 4
  store double* %a_vals, double** %a_vals.addr, align 4
  store i32 %num_of_rows, i32* %num_of_rows.addr, align 4
  store i32* %pos, i32** %pos.addr, align 4
  store i32* %crd, i32** %crd.addr, align 4
  store double* %B_vals, double** %B_vals.addr, align 4
  store double* %c_vals, double** %c_vals.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc10, %entry
  %0 = load i32, i32* %i, align 4
  %1 = load i32, i32* %num_of_rows.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end12

for.body:                                         ; preds = %for.cond
  store double 0.000000e+00, double* %res, align 8
  %2 = load i32*, i32** %pos.addr, align 4
  %3 = load i32, i32* %i, align 4
  %arrayidx = getelementptr inbounds i32, i32* %2, i32 %3
  %4 = load i32, i32* %arrayidx, align 4
  store i32 %4, i32* %jj, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %5 = load i32, i32* %jj, align 4
  %6 = load i32*, i32** %pos.addr, align 4
  %7 = load i32, i32* %i, align 4
  %add = add nsw i32 %7, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %6, i32 %add
  %8 = load i32, i32* %arrayidx2, align 4
  %cmp3 = icmp slt i32 %5, %8
  br i1 %cmp3, label %for.body4, label %for.end

for.body4:                                        ; preds = %for.cond1
  %9 = load double*, double** %B_vals.addr, align 4
  %10 = load i32, i32* %jj, align 4
  %arrayidx5 = getelementptr inbounds double, double* %9, i32 %10
  %11 = load double, double* %arrayidx5, align 4
  %12 = load double*, double** %c_vals.addr, align 4
  %13 = load i32*, i32** %crd.addr, align 4
  %14 = load i32, i32* %jj, align 4
  %arrayidx6 = getelementptr inbounds i32, i32* %13, i32 %14
  %15 = load i32, i32* %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds double, double* %12, i32 %15
  %16 = load double, double* %arrayidx7, align 4
  %mul = fmul double %11, %16
  %17 = load double, double* %res, align 8
  %add8 = fadd double %17, %mul
  store double %add8, double* %res, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body4
  %18 = load i32, i32* %jj, align 4
  %inc = add nsw i32 %18, 1
  store i32 %inc, i32* %jj, align 4
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  %19 = load double, double* %res, align 8
  %20 = load double*, double** %a_vals.addr, align 4
  %21 = load i32, i32* %i, align 4
  %arrayidx9 = getelementptr inbounds double, double* %20, i32 %21
  store double %19, double* %arrayidx9, align 4
  br label %for.inc10

for.inc10:                                        ; preds = %for.end
  %22 = load i32, i32* %i, align 4
  %inc11 = add nsw i32 %22, 1
  store i32 %inc11, i32* %i, align 4
  br label %for.cond

for.end12:                                        ; preds = %for.cond
  ret i32 0
}

attributes #0 = { noinline nounwind ssp "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+armv4t,+soft-float,+strict-align,-crypto,-d32,-dotprod,-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fp16,-fp16fml,-fp64,-fpregs,-fullfp16,-mve,-mve.fp,-neon,-thumb-mode,-vfp2,-vfp2sp,-vfp3,-vfp3d16,-vfp3d16sp,-vfp3sp,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp" "unsafe-fp-math"="false" "use-soft-float"="true" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{!"clang version 10.0.0 (git@github.com:kromancer/llvm-project.git d32170dbd5b0d54436537b6b75beaf44324e0c28)"}
