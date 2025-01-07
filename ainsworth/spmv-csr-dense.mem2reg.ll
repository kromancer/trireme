; ModuleID = 'spmv-csr-dense.ll'
source_filename = "spmv-csr-dense.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.20.0"

; Function Attrs: noinline nounwind ssp
define i32 @compute(double* noalias %a_vals, i32 %num_of_rows, i32* noalias %pos, i32* noalias %crd, double* noalias %B_vals, double* noalias %c_vals) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc10, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc11, %for.inc10 ]
  %cmp = icmp slt i32 %i.0, %num_of_rows
  br i1 %cmp, label %for.body, label %for.end12

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %pos, i32 %i.0
  %0 = load i32, i32* %arrayidx, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %res.0 = phi double [ 0.000000e+00, %for.body ], [ %add8, %for.inc ]
  %jj.0 = phi i32 [ %0, %for.body ], [ %inc, %for.inc ]
  %add = add nsw i32 %i.0, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %pos, i32 %add
  %1 = load i32, i32* %arrayidx2, align 4
  %cmp3 = icmp slt i32 %jj.0, %1
  br i1 %cmp3, label %for.body4, label %for.end

for.body4:                                        ; preds = %for.cond1
  %arrayidx5 = getelementptr inbounds double, double* %B_vals, i32 %jj.0
  %2 = load double, double* %arrayidx5, align 4
  %arrayidx6 = getelementptr inbounds i32, i32* %crd, i32 %jj.0
  %3 = load i32, i32* %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds double, double* %c_vals, i32 %3
  %4 = load double, double* %arrayidx7, align 4
  %mul = fmul double %2, %4
  %add8 = fadd double %res.0, %mul
  br label %for.inc

for.inc:                                          ; preds = %for.body4
  %inc = add nsw i32 %jj.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  %arrayidx9 = getelementptr inbounds double, double* %a_vals, i32 %i.0
  store double %res.0, double* %arrayidx9, align 4
  br label %for.inc10

for.inc10:                                        ; preds = %for.end
  %inc11 = add nsw i32 %i.0, 1
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
