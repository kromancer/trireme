; ModuleID = 'component-wise-mult-crs-coo.ll'
source_filename = "component-wise-mult-crs-coo.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.19.0"

; Function Attrs: noinline nounwind ssp
define void @compute(i32 %max_j, double* noalias %A_vals, i32* noalias %B_pos, i32* noalias %B_crd, double* noalias %B_vals, i32* noalias %C1_pos, i32* noalias %C1_crd, i32* noalias %C2_crd, double* noalias %C_vals) #0 {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %C1_pos, i32 0
  %0 = load i32, i32* %arrayidx, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %iC.0 = phi i32 [ %0, %entry ], [ %inc, %for.inc ]
  %arrayidx1 = getelementptr inbounds i32, i32* %C1_pos, i32 1
  %1 = load i32, i32* %arrayidx1, align 4
  %cmp = icmp slt i32 %iC.0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx2 = getelementptr inbounds i32, i32* %C1_crd, i32 %iC.0
  %2 = load i32, i32* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %B_pos, i32 %2
  %3 = load i32, i32* %arrayidx3, align 4
  %add = add nsw i32 %2, 1
  %arrayidx4 = getelementptr inbounds i32, i32* %B_pos, i32 %add
  %4 = load i32, i32* %arrayidx4, align 4
  %add5 = add nsw i32 %iC.0, 1
  br label %while.cond

while.cond:                                       ; preds = %if.end, %for.body
  %jB.0 = phi i32 [ %3, %for.body ], [ %add19, %if.end ]
  %jC.0 = phi i32 [ %iC.0, %for.body ], [ %add22, %if.end ]
  %cmp6 = icmp slt i32 %jB.0, %4
  br i1 %cmp6, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %while.cond
  %cmp7 = icmp slt i32 %jC.0, %add5
  br label %land.end

land.end:                                         ; preds = %land.rhs, %while.cond
  %5 = phi i1 [ false, %while.cond ], [ %cmp7, %land.rhs ]
  br i1 %5, label %while.body, label %while.end

while.body:                                       ; preds = %land.end
  %arrayidx8 = getelementptr inbounds i32, i32* %B_crd, i32 %jB.0
  %6 = load i32, i32* %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds i32, i32* %C2_crd, i32 %jC.0
  %7 = load i32, i32* %arrayidx9, align 4
  %cmp10 = icmp slt i32 %6, %7
  br i1 %cmp10, label %cond.true, label %cond.false

cond.true:                                        ; preds = %while.body
  br label %cond.end

cond.false:                                       ; preds = %while.body
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %6, %cond.true ], [ %7, %cond.false ]
  %cmp11 = icmp eq i32 %6, %cond
  br i1 %cmp11, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %cond.end
  %cmp12 = icmp eq i32 %7, %cond
  br i1 %cmp12, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %mul = mul nsw i32 %2, %max_j
  %add13 = add nsw i32 %mul, %cond
  %arrayidx14 = getelementptr inbounds double, double* %B_vals, i32 %jB.0
  %8 = load double, double* %arrayidx14, align 4
  %arrayidx15 = getelementptr inbounds double, double* %C_vals, i32 %jC.0
  %9 = load double, double* %arrayidx15, align 4
  %mul16 = fmul double %8, %9
  %arrayidx17 = getelementptr inbounds double, double* %A_vals, i32 %add13
  store double %mul16, double* %arrayidx17, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %cond.end
  %cmp18 = icmp eq i32 %6, %cond
  %conv = zext i1 %cmp18 to i32
  %add19 = add nsw i32 %jB.0, %conv
  %cmp20 = icmp eq i32 %7, %cond
  %conv21 = zext i1 %cmp20 to i32
  %add22 = add nsw i32 %jC.0, %conv21
  br label %while.cond

while.end:                                        ; preds = %land.end
  br label %for.inc

for.inc:                                          ; preds = %while.end
  %inc = add nsw i32 %iC.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

attributes #0 = { noinline nounwind ssp "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+armv4t,+soft-float,+strict-align,-crypto,-d32,-dotprod,-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fp16,-fp16fml,-fp64,-fpregs,-fullfp16,-mve,-mve.fp,-neon,-thumb-mode,-vfp2,-vfp2sp,-vfp3,-vfp3d16,-vfp3d16sp,-vfp3sp,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp" "unsafe-fp-math"="false" "use-soft-float"="true" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{!"clang version 10.0.0 (git@github.com:kromancer/llvm-project.git d32170dbd5b0d54436537b6b75beaf44324e0c28)"}
