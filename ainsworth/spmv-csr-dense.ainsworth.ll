; ModuleID = 'spmv-csr-dense.mem2reg.ll'
source_filename = "spmv-csr-dense.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.20.0"

; Function Attrs: nofree noinline norecurse nounwind ssp
define i32 @compute(double* noalias nocapture %a_vals, i32 %num_of_rows, i32* noalias nocapture readonly %pos, i32* noalias nocapture readonly %crd, double* noalias nocapture readonly %B_vals, double* noalias nocapture readonly %c_vals) local_unnamed_addr #0 {
entry:
  %cmp5 = icmp sgt i32 %num_of_rows, 0
  br i1 %cmp5, label %for.body.preheader, label %for.end12

for.body.preheader:                               ; preds = %entry
  %.pre = load i32, i32* %pos, align 4
  br label %for.body

for.body:                                         ; preds = %for.end, %for.body.preheader
  %0 = phi i32 [ %1, %for.end ], [ %.pre, %for.body.preheader ]
  %i.06 = phi i32 [ %add, %for.end ], [ 0, %for.body.preheader ]
  %add = add nuw nsw i32 %i.06, 1

  ; pos[i + 1]
  %arrayidx2 = getelementptr inbounds i32, i32* %pos, i32 %add
  %pos_i_plus_1 = load i32, i32* %arrayidx2, align 4

  %cmp31 = icmp slt i32 %0, %pos_i_plus_1
  br i1 %cmp31, label %for.body4.preheader, label %for.end

for.body4.preheader:                              ; preds = %for.body
  %jj_for_row_i_end = add i32 %pos_i_plus_1, -1
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader, %for.body4
  %jj.03 = phi i32 [ %inc, %for.body4 ], [ %0, %for.body4.preheader ]
  %res.02 = phi double [ %add8, %for.body4 ], [ 0.000000e+00, %for.body4.preheader ]
  %arrayidx5 = getelementptr inbounds double, double* %B_vals, i32 %jj.03
  %3 = load double, double* %arrayidx5, align 4
  %arrayidx6 = getelementptr inbounds i32, i32* %crd, i32 %jj.03

  ; crd[jj + 2 dist]
  %4 = add i32 %jj.03, 64
  %5 = getelementptr inbounds i32, i32* %crd, i32 %4
  %crd_jj_plus_2pd = bitcast i32* %5 to i8*

  %7 = load i32, i32* %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds double, double* %c_vals, i32 %7

  %jj_plus_pd = add i32 %jj.03, 32
  %9 = icmp slt i32 %jj_for_row_i_end, %jj_plus_pd
  %10 = select i1 %9, i32 %jj_for_row_i_end, i32 %jj_plus_pd
  %11 = getelementptr inbounds i32, i32* %crd, i32 %10
  %12 = load i32, i32* %11, align 4
  %13 = getelementptr inbounds double, double* %c_vals, i32 %12
  %14 = bitcast double* %13 to i8*

  %15 = load double, double* %arrayidx7, align 4
  %mul = fmul double %3, %15
  %add8 = fadd double %res.02, %mul
  %inc = add nsw i32 %jj.03, 1
  %cmp3 = icmp slt i32 %inc, %1

  call void @llvm.prefetch.p0i8(i8* %crd_jj_plus_2pd, i32 0, i32 3, i32 1)
  call void @llvm.prefetch.p0i8(i8* %14, i32 0, i32 3, i32 1)


  br i1 %cmp3, label %for.body4, label %for.end

for.end:                                          ; preds = %for.body4, %for.body
  %res.0.lcssa = phi double [ 0.000000e+00, %for.body ], [ %add8, %for.body4 ]
  %arrayidx9 = getelementptr inbounds double, double* %a_vals, i32 %i.06
  store double %res.0.lcssa, double* %arrayidx9, align 4
  %exitcond = icmp eq i32 %add, %num_of_rows
  br i1 %exitcond, label %for.end12, label %for.body

for.end12:                                        ; preds = %for.end, %entry
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
