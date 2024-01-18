; ModuleID = 'sddmm.ll'
source_filename = "sddmm.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.19.0"

; Function Attrs: noinline nounwind ssp
define void @compute(double* noalias %A_vals, i32* noalias %B1_pos, i32* noalias %B1_crd, i32* noalias %B2_pos, i32* noalias %B2_crd, double* noalias %B_vals, i32 %C2_dimension, double* noalias %C_vals, i32 %D1_dimension, i32 %D2_dimension, double* noalias %D_vals) #0 {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %B1_pos, i32 0
  %0 = load i32, i32* %arrayidx, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc28, %entry
  %jA.0 = phi i32 [ 0, %entry ], [ %jA.1, %for.inc28 ]
  %iB.0 = phi i32 [ %0, %entry ], [ %inc29, %for.inc28 ]
  %arrayidx1 = getelementptr inbounds i32, i32* %B1_pos, i32 1
  %1 = load i32, i32* %arrayidx1, align 4
  %cmp = icmp slt i32 %iB.0, %1
  br i1 %cmp, label %for.body, label %for.end30

for.body:                                         ; preds = %for.cond
  %arrayidx2 = getelementptr inbounds i32, i32* %B1_crd, i32 %iB.0
  %2 = load i32, i32* %arrayidx2, align 4
  br label %for.cond3

for.cond3:                                        ; preds = %for.inc25, %for.body
  %jA.1 = phi i32 [ %jA.0, %for.body ], [ %jA.2, %for.inc25 ]
  %k.0 = phi i32 [ 0, %for.body ], [ %inc26, %for.inc25 ]
  %cmp4 = icmp slt i32 %k.0, %D1_dimension
  br i1 %cmp4, label %for.body5, label %for.end27

for.body5:                                        ; preds = %for.cond3
  %mul = mul nsw i32 %2, %C2_dimension
  %add = add nsw i32 %mul, %k.0
  %arrayidx6 = getelementptr inbounds i32, i32* %B2_pos, i32 %iB.0
  %3 = load i32, i32* %arrayidx6, align 4
  br label %for.cond7

for.cond7:                                        ; preds = %for.inc, %for.body5
  %jA.2 = phi i32 [ %jA.1, %for.body5 ], [ %inc, %for.inc ]
  %jB.0 = phi i32 [ %3, %for.body5 ], [ %inc24, %for.inc ]
  %add8 = add nsw i32 %iB.0, 1
  %arrayidx9 = getelementptr inbounds i32, i32* %B2_pos, i32 %add8
  %4 = load i32, i32* %arrayidx9, align 4
  %cmp10 = icmp slt i32 %jB.0, %4
  br i1 %cmp10, label %for.body11, label %for.end

for.body11:                                       ; preds = %for.cond7
  %arrayidx12 = getelementptr inbounds i32, i32* %B2_crd, i32 %jB.0
  %5 = load i32, i32* %arrayidx12, align 4
  %arrayidx13 = getelementptr inbounds double, double* %A_vals, i32 %jA.2
  store double 0.000000e+00, double* %arrayidx13, align 4
  %mul14 = mul nsw i32 %k.0, %D2_dimension
  %add15 = add nsw i32 %mul14, %5
  %arrayidx16 = getelementptr inbounds double, double* %A_vals, i32 %jA.2
  %6 = load double, double* %arrayidx16, align 4
  %arrayidx17 = getelementptr inbounds double, double* %B_vals, i32 %jB.0
  %7 = load double, double* %arrayidx17, align 4
  %arrayidx18 = getelementptr inbounds double, double* %C_vals, i32 %add
  %8 = load double, double* %arrayidx18, align 4
  %mul19 = fmul double %7, %8
  %arrayidx20 = getelementptr inbounds double, double* %D_vals, i32 %add15
  %9 = load double, double* %arrayidx20, align 4
  %mul21 = fmul double %mul19, %9
  %add22 = fadd double %6, %mul21
  %arrayidx23 = getelementptr inbounds double, double* %A_vals, i32 %jA.2
  store double %add22, double* %arrayidx23, align 4
  %inc = add nsw i32 %jA.2, 1
  br label %for.inc

for.inc:                                          ; preds = %for.body11
  %inc24 = add nsw i32 %jB.0, 1
  br label %for.cond7

for.end:                                          ; preds = %for.cond7
  br label %for.inc25

for.inc25:                                        ; preds = %for.end
  %inc26 = add nsw i32 %k.0, 1
  br label %for.cond3

for.end27:                                        ; preds = %for.cond3
  br label %for.inc28

for.inc28:                                        ; preds = %for.end27
  %inc29 = add nsw i32 %iB.0, 1
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
