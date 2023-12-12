; ModuleID = 'spmv-csr-dense.mem2reg.ll'
source_filename = "spmv-csr-dense.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.19.0"

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
  %arrayidx2 = getelementptr inbounds i32, i32* %pos, i32 %add
  %1 = load i32, i32* %arrayidx2, align 4
  %cmp31 = icmp slt i32 %0, %1
  br i1 %cmp31, label %for.body4.preheader, label %for.end

for.body4.preheader:                              ; preds = %for.body
  %j_bound = add i32 %1, -1
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader, %for.body4
  %j = phi i32 [ %j_plus_1, %for.body4 ], [ %0, %for.body4.preheader ]
  %tja_val.02 = phi double [ %add8, %for.body4 ], [ 0.000000e+00, %for.body4.preheader ]

  ; compute address of crd[j]
  %crd_j_addr = getelementptr inbounds i32, i32* %crd, i32 %j

  ; Prefetch-related: compute address of crd[j + 2 * distance]
  %3 = add i32 %j, 64
  %4 = getelementptr inbounds i32, i32* %crd, i32 %3
  %crd_prefetch = bitcast i32* %4 to i8*

  ; load crd[j]
  %crd_j = load i32, i32* %crd_j_addr, align 4

  ; load B_vals[j]
  %6 = getelementptr inbounds double, double* %B_vals, i32 %j
  &B_vals_j = load double, double* %6, align 4

  ; compute address of c_vals[crd[j]]
  %c_vals_crd_j_addr = getelementptr inbounds double, double* %c_vals, i32 %crd_j

  ; Prefetch-related: compute min(j_bound, j + 32)
  %8 = add i32 %j, 32
  %9 = icmp slt i32 %j_bound, %8
  %dist = select i1 %9, i32 %j_bound, i32 %8

  ; Prefetch-related: load crd[dist]  
  %11 = getelementptr inbounds i32, i32* %crd, i32 %dist
  %crd_dist_addr = load i32, i32* %11, align 4

  ; Prefetch-related: compute address of c_vals[crd[dist]]
  %13 = getelementptr inbounds double, double* %c_vals, i32 %crd_dist_addr
  %c_vals_prefetch = bitcast double* %13 to i8*

  ; compute a[i] += B_vals[j] * c_vals[crd[j]]
  %15 = load double, double* %c_vals_crd_j_addr, align 4
  %mul = fmul double &B_vals_j, %15
  %add8 = fadd double %tja_val.02, %mul

  ; j loop termination condition
  %j_plus_1 = add nsw i32 %j, 1
  %cmp3 = icmp slt i32 %j_plus_1, %1
  
  call void @llvm.prefetch.p0i8(i8* %crd_prefetch, i32 0, i32 3, i32 1)
  call void @llvm.prefetch.p0i8(i8* %c_vals_prefetch, i32 0, i32 3, i32 1)
  
  br i1 %cmp3, label %for.body4, label %for.end

for.end:                                          ; preds = %for.body4, %for.body
  %tja_val.0.lcssa = phi double [ 0.000000e+00, %for.body ], [ %add8, %for.body4 ]
  %arrayidx9 = getelementptr inbounds double, double* %a_vals, i32 %i.06
  store double %tja_val.0.lcssa, double* %arrayidx9, align 4
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
