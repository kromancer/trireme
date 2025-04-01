; ModuleID = 'spmm-csr-dense.mem2reg.ll'
source_filename = "spmm-csr-dense.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.20.0"

; Function Attrs: nofree noinline norecurse nounwind ssp
define i32 @compute(i32 %A2_dimension, i32 %B1_dimension, i32 %C2_dimension, double* noalias nocapture %A_vals, double* noalias nocapture readonly %C_vals, i32* noalias nocapture readonly %B2_pos, i32* noalias nocapture readonly %B2_crd, double* noalias nocapture readonly %B_vals) local_unnamed_addr #0 {
entry:
  %cmp6 = icmp sgt i32 %B1_dimension, 0
  br i1 %cmp6, label %for.body.lr.ph, label %for.end23

for.body.lr.ph:                                   ; preds = %entry
  %cmp71 = icmp sgt i32 %C2_dimension, 0
  %.pre = load i32, i32* %B2_pos, align 4
  br label %for.body

for.cond.loopexit:                                ; preds = %for.inc18, %for.body
  %exitcond8 = icmp eq i32 %add, %B1_dimension
  br i1 %exitcond8, label %for.end23, label %for.body

for.body:                                         ; preds = %for.cond.loopexit, %for.body.lr.ph
  %0 = phi i32 [ %.pre, %for.body.lr.ph ], [ %1, %for.cond.loopexit ]
  %i.07 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.cond.loopexit ]
  %add = add nuw nsw i32 %i.07, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %B2_pos, i32 %add
  %1 = load i32, i32* %arrayidx2, align 4
  %cmp33 = icmp slt i32 %0, %1
  br i1 %cmp33, label %for.body4.lr.ph, label %for.cond.loopexit

for.body4.lr.ph:                                  ; preds = %for.body
  %mul = mul nsw i32 %i.07, %A2_dimension
  br label %for.body4

for.body4:                                        ; preds = %for.body4.lr.ph, %for.inc18
  %jB.04 = phi i32 [ %0, %for.body4.lr.ph ], [ %inc19, %for.inc18 ]
  br i1 %cmp71, label %for.body8.lr.ph, label %for.inc18

for.body8.lr.ph:                                  ; preds = %for.body4
  %arrayidx5 = getelementptr inbounds i32, i32* %B2_crd, i32 %jB.04
  %2 = load i32, i32* %arrayidx5, align 4
  %mul10 = mul nsw i32 %2, %C2_dimension
  %arrayidx13 = getelementptr inbounds double, double* %B_vals, i32 %jB.04
  %3 = load double, double* %arrayidx13, align 4
  br label %for.body8

for.body8:                                        ; preds = %for.body8, %for.body8.lr.ph
  %k.02 = phi i32 [ 0, %for.body8.lr.ph ], [ %inc, %for.body8 ]
  %add9 = add nsw i32 %k.02, %mul
  %add11 = add nsw i32 %k.02, %mul10
  %arrayidx12 = getelementptr inbounds double, double* %A_vals, i32 %add9
  %4 = load double, double* %arrayidx12, align 4
  %arrayidx14 = getelementptr inbounds double, double* %C_vals, i32 %add11
  %5 = load double, double* %arrayidx14, align 4
  %mul15 = fmul double %3, %5
  %add16 = fadd double %4, %mul15
  store double %add16, double* %arrayidx12, align 4
  %inc = add nuw nsw i32 %k.02, 1
  %exitcond = icmp eq i32 %inc, %C2_dimension
  br i1 %exitcond, label %for.inc18, label %for.body8

for.inc18:                                        ; preds = %for.body8, %for.body4
  %inc19 = add nsw i32 %jB.04, 1
  %cmp3 = icmp slt i32 %inc19, %1
  br i1 %cmp3, label %for.body4, label %for.cond.loopexit

for.end23:                                        ; preds = %for.cond.loopexit, %entry
  ret i32 0
}

attributes #0 = { nofree noinline norecurse nounwind ssp "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+armv4t,+soft-float,+strict-align,-crypto,-d32,-dotprod,-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fp16,-fp16fml,-fp64,-fpregs,-fullfp16,-mve,-mve.fp,-neon,-thumb-mode,-vfp2,-vfp2sp,-vfp3,-vfp3d16,-vfp3d16sp,-vfp3sp,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp" "unsafe-fp-math"="false" "use-soft-float"="true" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{!"clang version 10.0.0 (git@github.com:kromancer/llvm-project.git d32170dbd5b0d54436537b6b75beaf44324e0c28)"}
