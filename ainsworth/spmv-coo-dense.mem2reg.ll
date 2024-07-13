; ModuleID = 'spmv-coo-dense.ll'
source_filename = "spmv-coo-dense.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.19.0"

; Function Attrs: noinline nounwind ssp
define i32 @compute(double* noalias %a_vals, i32* noalias %B1_pos, i32* noalias %B1_crd, i32* noalias %B2_crd, double* noalias %B_vals, double* noalias %c_vals) #0 {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %B1_pos, i32 0
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %B1_pos, i32 1
  %1 = load i32, i32* %arrayidx1, align 4
  br label %while.cond

while.cond:                                       ; preds = %for.end, %entry
  %iB.0 = phi i32 [ %0, %entry ], [ %B1_segend.0, %for.end ]
  %cmp = icmp slt i32 %iB.0, %1
  br i1 %cmp, label %while.body, label %while.end15

while.body:                                       ; preds = %while.cond
  %arrayidx2 = getelementptr inbounds i32, i32* %B1_crd, i32 %iB.0
  %2 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %iB.0, 1
  br label %while.cond3

while.cond3:                                      ; preds = %while.body7, %while.body
  %B1_segend.0 = phi i32 [ %add, %while.body ], [ %inc, %while.body7 ]
  %cmp4 = icmp slt i32 %B1_segend.0, %1
  br i1 %cmp4, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %while.cond3
  %arrayidx5 = getelementptr inbounds i32, i32* %B1_crd, i32 %B1_segend.0
  %3 = load i32, i32* %arrayidx5, align 4
  %cmp6 = icmp eq i32 %3, %2
  br label %land.end

land.end:                                         ; preds = %land.rhs, %while.cond3
  %4 = phi i1 [ false, %while.cond3 ], [ %cmp6, %land.rhs ]
  br i1 %4, label %while.body7, label %while.end

while.body7:                                      ; preds = %land.end
  %inc = add nsw i32 %B1_segend.0, 1
  br label %while.cond3

while.end:                                        ; preds = %land.end
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %while.end
  %tja_val.0 = phi double [ 0.000000e+00, %while.end ], [ %add12, %for.inc ]
  %jB.0 = phi i32 [ %iB.0, %while.end ], [ %inc13, %for.inc ]
  %cmp8 = icmp slt i32 %jB.0, %B1_segend.0
  br i1 %cmp8, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx9 = getelementptr inbounds i32, i32* %B2_crd, i32 %jB.0
  %5 = load i32, i32* %arrayidx9, align 4
  %arrayidx10 = getelementptr inbounds double, double* %B_vals, i32 %jB.0
  %6 = load double, double* %arrayidx10, align 4
  %arrayidx11 = getelementptr inbounds double, double* %c_vals, i32 %5
  %7 = load double, double* %arrayidx11, align 4
  %mul = fmul double %6, %7
  %add12 = fadd double %tja_val.0, %mul
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc13 = add nsw i32 %jB.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %arrayidx14 = getelementptr inbounds double, double* %a_vals, i32 %2
  store double %tja_val.0, double* %arrayidx14, align 4
  br label %while.cond

while.end15:                                      ; preds = %while.cond
  ret i32 0
}

attributes #0 = { noinline nounwind ssp "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+armv4t,+soft-float,+strict-align,-crypto,-d32,-dotprod,-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fp16,-fp16fml,-fp64,-fpregs,-fullfp16,-mve,-mve.fp,-neon,-thumb-mode,-vfp2,-vfp2sp,-vfp3,-vfp3d16,-vfp3d16sp,-vfp3sp,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp" "unsafe-fp-math"="false" "use-soft-float"="true" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{!"clang version 10.0.0 (git@github.com:kromancer/llvm-project.git d32170dbd5b0d54436537b6b75beaf44324e0c28)"}
