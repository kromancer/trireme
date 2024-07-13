; ModuleID = 'spmv-coo-dense.mem2reg.ll'
source_filename = "spmv-coo-dense.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.19.0"

; Function Attrs: nofree noinline norecurse nounwind ssp
define i32 @compute(double* noalias nocapture %a_vals, i32* noalias nocapture readonly %B1_pos, i32* noalias nocapture readonly %B1_crd, i32* noalias nocapture readonly %B2_crd, double* noalias nocapture readonly %B_vals, double* noalias nocapture readonly %c_vals) local_unnamed_addr #0 {
entry:
  %0 = load i32, i32* %B1_pos, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %B1_pos, i32 1
  %1 = load i32, i32* %arrayidx1, align 4
  %cmp5 = icmp slt i32 %0, %1
  br i1 %cmp5, label %while.body, label %while.end15

while.body:                                       ; preds = %entry, %for.end
  %iB.06 = phi i32 [ %B1_segend.0, %for.end ], [ %0, %entry ]
  %arrayidx2 = getelementptr inbounds i32, i32* %B1_crd, i32 %iB.06
  %2 = load i32, i32* %arrayidx2, align 4
  br label %while.cond3

while.cond3:                                      ; preds = %land.rhs, %while.body
  %B1_segend.0.in = phi i32 [ %iB.06, %while.body ], [ %B1_segend.0, %land.rhs ]
  %B1_segend.0 = add nsw i32 %B1_segend.0.in, 1
  %cmp4 = icmp slt i32 %B1_segend.0, %1
  br i1 %cmp4, label %land.rhs, label %while.end

land.rhs:                                         ; preds = %while.cond3
  %arrayidx5 = getelementptr inbounds i32, i32* %B1_crd, i32 %B1_segend.0
  %3 = load i32, i32* %arrayidx5, align 4
  %cmp6 = icmp eq i32 %3, %2
  br i1 %cmp6, label %while.cond3, label %while.end

while.end:                                        ; preds = %while.cond3, %land.rhs
  %cmp81 = icmp sgt i32 %iB.06, %B1_segend.0.in
  br i1 %cmp81, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %while.end
  %4 = add i32 %B1_segend.0.in, -1
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %jB.03 = phi i32 [ %inc13, %for.body ], [ %iB.06, %for.body.preheader ]
  %tja_val.02 = phi double [ %add12, %for.body ], [ 0.000000e+00, %for.body.preheader ]
  %arrayidx9 = getelementptr inbounds i32, i32* %B2_crd, i32 %jB.03

  ; computes the address of B2_crd[jB + 2 * distance], which is prefetched in line 74
  %5 = add i32 %jB.03, 64
  %6 = getelementptr inbounds i32, i32* %B2_crd, i32 %5
  %7 = bitcast i32* %6 to i8*

  %8 = load i32, i32* %arrayidx9, align 4
  %arrayidx10 = getelementptr inbounds double, double* %B_vals, i32 %jB.03
  %9 = load double, double* %arrayidx10, align 4
  %arrayidx11 = getelementptr inbounds double, double* %c_vals, i32 %8

  ; computes min(jB + distance, B1_segend)
  %10 = add i32 %jB.03, 32
  %11 = icmp slt i32 %4, %10
  %12 = select i1 %11, i32 %4, i32 %10

  ; loads B2_crd[min]
  %13 = getelementptr inbounds i32, i32* %B2_crd, i32 %12
  %14 = load i32, i32* %13, align 4

  ; computes the address of c_vals[B2_cord[min]] which is prefetched in line 75
  %15 = getelementptr inbounds double, double* %c_vals, i32 %14
  %16 = bitcast double* %15 to i8*

  %17 = load double, double* %arrayidx11, align 4
  %mul = fmul double %9, %17
  %add12 = fadd double %tja_val.02, %mul
  %inc13 = add nsw i32 %jB.03, 1
  %cmp8 = icmp slt i32 %jB.03, %B1_segend.0.in
  call void @llvm.prefetch.p0i8(i8* %7, i32 0, i32 3, i32 1)
  call void @llvm.prefetch.p0i8(i8* %16, i32 0, i32 3, i32 1)
  br i1 %cmp8, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %while.end
  %tja_val.0.lcssa = phi double [ 0.000000e+00, %while.end ], [ %add12, %for.body ]
  %arrayidx14 = getelementptr inbounds double, double* %a_vals, i32 %2
  store double %tja_val.0.lcssa, double* %arrayidx14, align 4
  br i1 %cmp4, label %while.body, label %while.end15

while.end15:                                      ; preds = %for.end, %entry
  ret i32 0
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
