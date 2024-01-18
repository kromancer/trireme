; ModuleID = 'sddmm.mem2reg.ll'
source_filename = "sddmm.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.19.0"

; Function Attrs: nofree noinline norecurse nounwind ssp
define void @compute(double* noalias nocapture %A_vals, i32* noalias nocapture readonly %B1_pos, i32* noalias nocapture readonly %B1_crd, i32* noalias nocapture readonly %B2_pos, i32* noalias nocapture readonly %B2_crd, double* noalias nocapture readonly %B_vals, i32 %C2_dimension, double* noalias nocapture readonly %C_vals, i32 %D1_dimension, i32 %D2_dimension, double* noalias nocapture readonly %D_vals) local_unnamed_addr #0 {
entry:
  %0 = load i32, i32* %B1_pos, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %B1_pos, i32 1
  %1 = load i32, i32* %arrayidx1, align 4
  %cmp9 = icmp slt i32 %0, %1
  br i1 %cmp9, label %for.body.lr.ph, label %for.end30

for.body.lr.ph:                                   ; preds = %entry
  %cmp44 = icmp sgt i32 %D1_dimension, 0
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.inc28
  %iB.011 = phi i32 [ %0, %for.body.lr.ph ], [ %inc29.pre-phi, %for.inc28 ]
  %jA.010 = phi i32 [ 0, %for.body.lr.ph ], [ %jA.1.lcssa, %for.inc28 ]
  br i1 %cmp44, label %for.body5.lr.ph, label %for.body.for.inc28_crit_edge

for.body.for.inc28_crit_edge:                     ; preds = %for.body
  %.pre = add nsw i32 %iB.011, 1
  br label %for.inc28

for.body5.lr.ph:                                  ; preds = %for.body
  %arrayidx2 = getelementptr inbounds i32, i32* %B1_crd, i32 %iB.011
  %2 = load i32, i32* %arrayidx2, align 4
  %mul = mul nsw i32 %2, %C2_dimension
  %arrayidx6 = getelementptr inbounds i32, i32* %B2_pos, i32 %iB.011
  %3 = load i32, i32* %arrayidx6, align 4
  %add8 = add nsw i32 %iB.011, 1
  %arrayidx9 = getelementptr inbounds i32, i32* %B2_pos, i32 %add8
  %4 = load i32, i32* %arrayidx9, align 4
  %cmp101 = icmp slt i32 %3, %4
  %5 = add i32 %4, -1
  br label %for.body5

for.body5:                                        ; preds = %for.inc25, %for.body5.lr.ph
  %k.06 = phi i32 [ 0, %for.body5.lr.ph ], [ %inc26, %for.inc25 ]
  %jA.15 = phi i32 [ %jA.010, %for.body5.lr.ph ], [ %jA.2.lcssa, %for.inc25 ]
  br i1 %cmp101, label %for.body11.lr.ph, label %for.inc25

for.body11.lr.ph:                                 ; preds = %for.body5
  %add = add nsw i32 %k.06, %mul
  %mul14 = mul nsw i32 %k.06, %D2_dimension
  %arrayidx18 = getelementptr inbounds double, double* %C_vals, i32 %add
  %6 = load double, double* %arrayidx18, align 4
  br label %for.body11

for.body11:                                       ; preds = %for.body11.lr.ph, %for.body11
  %jB.03 = phi i32 [ %3, %for.body11.lr.ph ], [ %inc24, %for.body11 ]
  %jA.22 = phi i32 [ %jA.15, %for.body11.lr.ph ], [ %inc, %for.body11 ]
  %arrayidx12 = getelementptr inbounds i32, i32* %B2_crd, i32 %jB.03
  %7 = add i32 %jB.03, 64
  %8 = getelementptr inbounds i32, i32* %B2_crd, i32 %7
  %9 = bitcast i32* %8 to i8*
  %10 = load i32, i32* %arrayidx12, align 4
  %arrayidx13 = getelementptr inbounds double, double* %A_vals, i32 %jA.22
  %add15 = add nsw i32 %10, %mul14
  %arrayidx17 = getelementptr inbounds double, double* %B_vals, i32 %jB.03
  %11 = load double, double* %arrayidx17, align 4
  %mul19 = fmul double %11, %6
  %arrayidx20 = getelementptr inbounds double, double* %D_vals, i32 %add15
  %12 = add i32 %jB.03, 32
  %13 = icmp slt i32 %5, %12
  %14 = select i1 %13, i32 %5, i32 %12
  %15 = getelementptr inbounds i32, i32* %B2_crd, i32 %14
  %16 = load i32, i32* %15, align 4
  %17 = add nsw i32 %16, %mul14
  %18 = getelementptr inbounds double, double* %D_vals, i32 %17
  %19 = bitcast double* %18 to i8*
  %20 = load double, double* %arrayidx20, align 4
  %mul21 = fmul double %mul19, %20
  %add22 = fadd double %mul21, 0.000000e+00
  store double %add22, double* %arrayidx13, align 4
  %inc = add nsw i32 %jA.22, 1
  %inc24 = add nsw i32 %jB.03, 1
  %cmp10 = icmp slt i32 %inc24, %4
  call void @llvm.prefetch.p0i8(i8* %9, i32 0, i32 3, i32 1)
  call void @llvm.prefetch.p0i8(i8* %19, i32 0, i32 3, i32 1)
  br i1 %cmp10, label %for.body11, label %for.inc25

for.inc25:                                        ; preds = %for.body11, %for.body5
  %jA.2.lcssa = phi i32 [ %jA.15, %for.body5 ], [ %inc, %for.body11 ]
  %inc26 = add nuw nsw i32 %k.06, 1
  %exitcond = icmp eq i32 %inc26, %D1_dimension
  br i1 %exitcond, label %for.inc28, label %for.body5

for.inc28:                                        ; preds = %for.inc25, %for.body.for.inc28_crit_edge
  %inc29.pre-phi = phi i32 [ %.pre, %for.body.for.inc28_crit_edge ], [ %add8, %for.inc25 ]
  %jA.1.lcssa = phi i32 [ %jA.010, %for.body.for.inc28_crit_edge ], [ %jA.2.lcssa, %for.inc25 ]
  %cmp = icmp slt i32 %inc29.pre-phi, %1
  br i1 %cmp, label %for.body, label %for.end30

for.end30:                                        ; preds = %for.inc28, %entry
  ret void
}

; Function Attrs: inaccessiblemem_or_argmemonly nounwind willreturn
declare void @llvm.prefetch.p0i8(i8* nocapture readonly, i32 immarg, i32 immarg, i32) #1

attributes #0 = { nofree noinline norecurse nounwind ssp "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+armv4t,+soft-float,+strict-align,-crypto,-d32,-dotprod,-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fp16,-fp16fml,-fp64,-fpregs,-fullfp16,-mve,-mve.fp,-neon,-thumb-mode,-vfp2,-vfp2sp,-vfp3,-vfp3d16,-vfp3d16sp,-vfp3sp,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #1 = { inaccessiblemem_or_argmemonly nounwind willreturn }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{!"clang version 10.0.0 (git@github.com:kromancer/llvm-project.git d32170dbd5b0d54436537b6b75beaf44324e0c28)"}
