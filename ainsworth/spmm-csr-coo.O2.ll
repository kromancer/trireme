; ModuleID = 'spmm-csr-coo.mem2reg.ll'
source_filename = "spmm-csr-coo.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.19.0"

; Function Attrs: nofree noinline norecurse nounwind ssp
define void @compute(i32 %B1_dimension, i32 %A2_dimension, double* noalias nocapture %A_vals, i32* noalias nocapture readonly %B2_pos, i32* noalias nocapture readonly %B2_crd, double* noalias nocapture readonly %B_vals, i32* noalias nocapture readonly %C1_pos, i32* noalias nocapture readonly %C1_crd, i32* noalias nocapture readonly %C2_crd, double* noalias nocapture readonly %C_vals) local_unnamed_addr #0 {
entry:
  %cmp14 = icmp sgt i32 %B1_dimension, 0
  br i1 %cmp14, label %for.body.lr.ph, label %for.end36

for.body.lr.ph:                                   ; preds = %entry
  %0 = load i32, i32* %C1_pos, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %C1_pos, i32 1
  %1 = load i32, i32* %arrayidx3, align 4
  %cmp57 = icmp slt i32 %0, %1
  %.pre = load i32, i32* %B2_pos, align 4
  br label %for.body

for.cond.loopexit:                                ; preds = %if.end, %for.body
  %exitcond17 = icmp eq i32 %add, %B1_dimension
  br i1 %exitcond17, label %for.end36, label %for.body

for.body:                                         ; preds = %for.cond.loopexit, %for.body.lr.ph
  %2 = phi i32 [ %.pre, %for.body.lr.ph ], [ %3, %for.cond.loopexit ]
  %i.015 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.cond.loopexit ]
  %add = add nuw nsw i32 %i.015, 1
  %arrayidx1 = getelementptr inbounds i32, i32* %B2_pos, i32 %add
  %3 = load i32, i32* %arrayidx1, align 4
  %cmp46 = icmp slt i32 %2, %3
  %spec.select8 = and i1 %cmp46, %cmp57
  br i1 %spec.select8, label %while.body.lr.ph, label %for.cond.loopexit

while.body.lr.ph:                                 ; preds = %for.body
  %mul = mul nsw i32 %i.015, %A2_dimension
  br label %while.body

while.body:                                       ; preds = %while.body.lr.ph, %if.end
  %kC.010 = phi i32 [ %0, %while.body.lr.ph ], [ %add32, %if.end ]
  %kB.09 = phi i32 [ %2, %while.body.lr.ph ], [ %add31, %if.end ]
  %arrayidx6 = getelementptr inbounds i32, i32* %B2_crd, i32 %kB.09
  %4 = load i32, i32* %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds i32, i32* %C1_crd, i32 %kC.010
  %5 = load i32, i32* %arrayidx7, align 4
  %cmp8 = icmp slt i32 %4, %5
  %. = select i1 %cmp8, i32 %4, i32 %5
  %cmp101 = icmp sge i32 %kC.010, %1
  %cmp1319 = icmp sgt i32 %5, %4
  %or.cond22 = or i1 %cmp101, %cmp1319
  br i1 %or.cond22, label %while.end, label %while.body15

while.body15:                                     ; preds = %while.body, %while.body15.land.rhs11_crit_edge
  %C1_segend.0220 = phi i32 [ %inc, %while.body15.land.rhs11_crit_edge ], [ %kC.010, %while.body ]
  %inc = add i32 %C1_segend.0220, 1
  %exitcond = icmp eq i32 %inc, %1
  br i1 %exitcond, label %while.end, label %while.body15.land.rhs11_crit_edge

while.body15.land.rhs11_crit_edge:                ; preds = %while.body15
  %arrayidx12.phi.trans.insert = getelementptr inbounds i32, i32* %C1_crd, i32 %inc
  %.pre18 = load i32, i32* %arrayidx12.phi.trans.insert, align 4
  %cmp13 = icmp eq i32 %.pre18, %.
  br i1 %cmp13, label %while.body15, label %while.end

while.end:                                        ; preds = %while.body15, %while.body15.land.rhs11_crit_edge, %while.body
  %C1_segend.0.lcssa = phi i32 [ %kC.010, %while.body ], [ %inc, %while.body15.land.rhs11_crit_edge ], [ %1, %while.body15 ]
  %cmp16 = icmp sge i32 %5, %4
  %6 = icmp eq i32 %5, %4
  %cmp194 = icmp slt i32 %kC.010, %C1_segend.0.lcssa
  %or.cond = and i1 %6, %cmp194
  br i1 %or.cond, label %for.body20.lr.ph, label %if.end

for.body20.lr.ph:                                 ; preds = %while.end
  %arrayidx24 = getelementptr inbounds double, double* %B_vals, i32 %kB.09
  %7 = load double, double* %arrayidx24, align 4
  br label %for.body20

for.body20:                                       ; preds = %for.body20, %for.body20.lr.ph
  %jC.05 = phi i32 [ %kC.010, %for.body20.lr.ph ], [ %inc29, %for.body20 ]
  %arrayidx21 = getelementptr inbounds i32, i32* %C2_crd, i32 %jC.05
  %8 = load i32, i32* %arrayidx21, align 4
  %add22 = add nsw i32 %8, %mul
  %arrayidx23 = getelementptr inbounds double, double* %A_vals, i32 %add22
  %9 = load double, double* %arrayidx23, align 4
  %arrayidx25 = getelementptr inbounds double, double* %C_vals, i32 %jC.05
  %10 = load double, double* %arrayidx25, align 4
  %mul26 = fmul double %7, %10
  %add27 = fadd double %9, %mul26
  store double %add27, double* %arrayidx23, align 4
  %inc29 = add nsw i32 %jC.05, 1
  %exitcond16 = icmp eq i32 %inc29, %C1_segend.0.lcssa
  br i1 %exitcond16, label %if.end, label %for.body20

if.end:                                           ; preds = %for.body20, %while.end
  %conv = zext i1 %cmp16 to i32
  %add31 = add nsw i32 %kB.09, %conv
  %add32 = add nsw i32 %C1_segend.0.lcssa, %kC.010
  %cmp4 = icmp slt i32 %add31, %3
  %cmp5 = icmp slt i32 %add32, %1
  %spec.select = and i1 %cmp4, %cmp5
  br i1 %spec.select, label %while.body, label %for.cond.loopexit

for.end36:                                        ; preds = %for.cond.loopexit, %entry
  ret void
}

attributes #0 = { nofree noinline norecurse nounwind ssp "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+armv4t,+soft-float,+strict-align,-crypto,-d32,-dotprod,-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fp16,-fp16fml,-fp64,-fpregs,-fullfp16,-mve,-mve.fp,-neon,-thumb-mode,-vfp2,-vfp2sp,-vfp3,-vfp3d16,-vfp3d16sp,-vfp3sp,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp" "unsafe-fp-math"="false" "use-soft-float"="true" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{!"clang version 10.0.0 (git@github.com:kromancer/llvm-project.git d32170dbd5b0d54436537b6b75beaf44324e0c28)"}
