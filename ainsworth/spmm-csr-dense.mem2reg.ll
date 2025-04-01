; ModuleID = 'spmm-csr-dense.ll'
source_filename = "spmm-csr-dense.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.20.0"

; Function Attrs: noinline nounwind ssp
define i32 @compute(i32 %A2_dimension, i32 %B1_dimension, i32 %C2_dimension, double* noalias %A_vals, double* noalias %C_vals, i32* noalias %B2_pos, i32* noalias %B2_crd, double* noalias %B_vals) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc21, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc22, %for.inc21 ]
  %cmp = icmp slt i32 %i.0, %B1_dimension
  br i1 %cmp, label %for.body, label %for.end23

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %B2_pos, i32 %i.0
  %0 = load i32, i32* %arrayidx, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc18, %for.body
  %jB.0 = phi i32 [ %0, %for.body ], [ %inc19, %for.inc18 ]
  %add = add nsw i32 %i.0, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %B2_pos, i32 %add
  %1 = load i32, i32* %arrayidx2, align 4
  %cmp3 = icmp slt i32 %jB.0, %1
  br i1 %cmp3, label %for.body4, label %for.end20

for.body4:                                        ; preds = %for.cond1
  %arrayidx5 = getelementptr inbounds i32, i32* %B2_crd, i32 %jB.0
  %2 = load i32, i32* %arrayidx5, align 4
  br label %for.cond6

for.cond6:                                        ; preds = %for.inc, %for.body4
  %k.0 = phi i32 [ 0, %for.body4 ], [ %inc, %for.inc ]
  %cmp7 = icmp slt i32 %k.0, %C2_dimension
  br i1 %cmp7, label %for.body8, label %for.end

for.body8:                                        ; preds = %for.cond6
  %mul = mul nsw i32 %i.0, %A2_dimension
  %add9 = add nsw i32 %mul, %k.0
  %mul10 = mul nsw i32 %2, %C2_dimension
  %add11 = add nsw i32 %mul10, %k.0
  %arrayidx12 = getelementptr inbounds double, double* %A_vals, i32 %add9
  %3 = load double, double* %arrayidx12, align 4
  %arrayidx13 = getelementptr inbounds double, double* %B_vals, i32 %jB.0
  %4 = load double, double* %arrayidx13, align 4
  %arrayidx14 = getelementptr inbounds double, double* %C_vals, i32 %add11
  %5 = load double, double* %arrayidx14, align 4
  %mul15 = fmul double %4, %5
  %add16 = fadd double %3, %mul15
  %arrayidx17 = getelementptr inbounds double, double* %A_vals, i32 %add9
  store double %add16, double* %arrayidx17, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body8
  %inc = add nsw i32 %k.0, 1
  br label %for.cond6

for.end:                                          ; preds = %for.cond6
  br label %for.inc18

for.inc18:                                        ; preds = %for.end
  %inc19 = add nsw i32 %jB.0, 1
  br label %for.cond1

for.end20:                                        ; preds = %for.cond1
  br label %for.inc21

for.inc21:                                        ; preds = %for.end20
  %inc22 = add nsw i32 %i.0, 1
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
