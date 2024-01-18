; ModuleID = 'component-wise-mult-crs-coo.mem2reg.ll'
source_filename = "component-wise-mult-crs-coo.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.19.0"

; Function Attrs: nofree noinline norecurse nounwind ssp
define void @compute(i32 %max_j, double* noalias nocapture %A_vals, i32* noalias nocapture readonly %B_pos, i32* noalias nocapture readonly %B_crd, double* noalias nocapture readonly %B_vals, i32* noalias nocapture readonly %C1_pos, i32* noalias nocapture readonly %C1_crd, i32* noalias nocapture readonly %C2_crd, double* noalias nocapture readonly %C_vals) local_unnamed_addr #0 {
entry:
  %0 = load i32, i32* %C1_pos, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C1_pos, i32 1
  %1 = load i32, i32* %arrayidx1, align 4
  %cmp5 = icmp slt i32 %0, %1
  br i1 %cmp5, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %2 = add i32 %1, -1
  %3 = add i32 %1, -1
  %4 = add i32 %1, -1
  %5 = add i32 %1, -1
  br label %for.body

for.cond.loopexit:                                ; preds = %if.end, %for.body
  %cmp = icmp slt i32 %add5, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.body.preheader, %for.cond.loopexit
  %iC.06 = phi i32 [ %add5, %for.cond.loopexit ], [ %0, %for.body.preheader ]
  %arrayidx2 = getelementptr inbounds i32, i32* %C1_crd, i32 %iC.06
  %6 = add i32 %iC.06, 64
  %7 = getelementptr inbounds i32, i32* %C1_crd, i32 %6
  %8 = bitcast i32* %7 to i8*
  %9 = load i32, i32* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %B_pos, i32 %9
  %10 = add i32 %iC.06, 48
  %11 = icmp slt i32 %2, %10
  %12 = select i1 %11, i32 %2, i32 %10
  %13 = getelementptr inbounds i32, i32* %C1_crd, i32 %12
  %14 = load i32, i32* %13, align 4
  %15 = getelementptr inbounds i32, i32* %B_pos, i32 %14
  %16 = bitcast i32* %15 to i8*
  %17 = load i32, i32* %arrayidx3, align 4
  %add = add nsw i32 %9, 1
  %arrayidx4 = getelementptr inbounds i32, i32* %B_pos, i32 %add
  %18 = add i32 %iC.06, 32
  %19 = icmp slt i32 %3, %18
  %20 = select i1 %19, i32 %3, i32 %18
  %21 = getelementptr inbounds i32, i32* %C1_crd, i32 %20
  %22 = load i32, i32* %21, align 4
  %23 = add nsw i32 %22, 1
  %24 = getelementptr inbounds i32, i32* %B_pos, i32 %23
  %25 = bitcast i32* %24 to i8*
  %26 = load i32, i32* %arrayidx4, align 4
  %add5 = add nsw i32 %iC.06, 1
  %cmp61 = icmp slt i32 %17, %26
  call void @llvm.prefetch.p0i8(i8* %8, i32 0, i32 3, i32 1)
  call void @llvm.prefetch.p0i8(i8* %16, i32 0, i32 3, i32 1)
  call void @llvm.prefetch.p0i8(i8* %25, i32 0, i32 3, i32 1)
  br i1 %cmp61, label %while.body.lr.ph, label %for.cond.loopexit

while.body.lr.ph:                                 ; preds = %for.body
  %mul = mul nsw i32 %9, %max_j
  %27 = add i32 %iC.06, 21
  %28 = icmp slt i32 %4, %27
  %29 = select i1 %28, i32 %4, i32 %27
  %30 = getelementptr inbounds i32, i32* %C1_crd, i32 %29
  %31 = load i32, i32* %30, align 4
  %32 = getelementptr inbounds i32, i32* %B_pos, i32 %31
  %33 = load i32, i32* %32, align 4
  %34 = getelementptr inbounds i32, i32* %B_crd, i32 %33
  %35 = bitcast i32* %34 to i8*
  call void @llvm.prefetch.p0i8(i8* %35, i32 0, i32 3, i32 1)
  %36 = add i32 %iC.06, 21
  %37 = icmp slt i32 %5, %36
  %38 = select i1 %37, i32 %5, i32 %36
  %39 = getelementptr inbounds i32, i32* %C1_crd, i32 %38
  %40 = load i32, i32* %39, align 4
  %41 = getelementptr inbounds i32, i32* %B_pos, i32 %40
  %42 = load i32, i32* %41, align 4
  %43 = getelementptr inbounds double, double* %B_vals, i32 %42
  %44 = bitcast double* %43 to i8*
  call void @llvm.prefetch.p0i8(i8* %44, i32 0, i32 3, i32 1)
  br label %while.body

while.body:                                       ; preds = %while.body.lr.ph, %if.end
  %jC.04 = phi i32 [ %iC.06, %while.body.lr.ph ], [ %add22, %if.end ]
  %jB.03 = phi i32 [ %17, %while.body.lr.ph ], [ %add19, %if.end ]
  %arrayidx8 = getelementptr inbounds i32, i32* %B_crd, i32 %jB.03
  %45 = load i32, i32* %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds i32, i32* %C2_crd, i32 %jC.04
  %46 = load i32, i32* %arrayidx9, align 4
  %cmp11 = icmp sge i32 %46, %45
  %cmp12 = icmp sle i32 %46, %45
  %47 = icmp eq i32 %46, %45
  br i1 %47, label %if.then, label %if.end

if.then:                                          ; preds = %while.body
  %add13 = add nsw i32 %45, %mul
  %arrayidx14 = getelementptr inbounds double, double* %B_vals, i32 %jB.03
  %48 = load double, double* %arrayidx14, align 4
  %arrayidx15 = getelementptr inbounds double, double* %C_vals, i32 %jC.04
  %49 = load double, double* %arrayidx15, align 4
  %mul16 = fmul double %48, %49
  %arrayidx17 = getelementptr inbounds double, double* %A_vals, i32 %add13
  store double %mul16, double* %arrayidx17, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %while.body
  %conv = zext i1 %cmp11 to i32
  %add19 = add nsw i32 %jB.03, %conv
  %conv21 = zext i1 %cmp12 to i32
  %add22 = add nsw i32 %jC.04, %conv21
  %cmp6 = icmp slt i32 %add19, %26
  %cmp7 = icmp sle i32 %add22, %iC.06
  %spec.select = and i1 %cmp6, %cmp7
  br i1 %spec.select, label %while.body, label %for.cond.loopexit

for.end:                                          ; preds = %for.cond.loopexit, %entry
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
