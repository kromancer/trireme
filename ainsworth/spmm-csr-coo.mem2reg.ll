; ModuleID = 'spmm-csr-coo.ll'
source_filename = "spmm-csr-coo.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.19.0"

; Function Attrs: noinline nounwind ssp
define void @compute(i32 %B1_dimension, i32 %A2_dimension, double* noalias %A_vals, i32* noalias %B2_pos, i32* noalias %B2_crd, double* noalias %B_vals, i32* noalias %C1_pos, i32* noalias %C1_crd, i32* noalias %C2_crd, double* noalias %C_vals) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc34, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc35, %for.inc34 ]
  %cmp = icmp slt i32 %i.0, %B1_dimension
  br i1 %cmp, label %for.body, label %for.end36

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %B2_pos, i32 %i.0
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %i.0, 1
  %arrayidx1 = getelementptr inbounds i32, i32* %B2_pos, i32 %add
  %1 = load i32, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %C1_pos, i32 0
  %2 = load i32, i32* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %C1_pos, i32 1
  %3 = load i32, i32* %arrayidx3, align 4
  br label %while.cond

while.cond:                                       ; preds = %if.end, %for.body
  %kB.0 = phi i32 [ %0, %for.body ], [ %add31, %if.end ]
  %kC.0 = phi i32 [ %2, %for.body ], [ %add32, %if.end ]
  %cmp4 = icmp slt i32 %kB.0, %1
  br i1 %cmp4, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %while.cond
  %cmp5 = icmp slt i32 %kC.0, %3
  br label %land.end

land.end:                                         ; preds = %land.rhs, %while.cond
  %4 = phi i1 [ false, %while.cond ], [ %cmp5, %land.rhs ]
  br i1 %4, label %while.body, label %while.end33

while.body:                                       ; preds = %land.end
  %arrayidx6 = getelementptr inbounds i32, i32* %B2_crd, i32 %kB.0
  %5 = load i32, i32* %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds i32, i32* %C1_crd, i32 %kC.0
  %6 = load i32, i32* %arrayidx7, align 4
  %cmp8 = icmp slt i32 %5, %6
  br i1 %cmp8, label %cond.true, label %cond.false

cond.true:                                        ; preds = %while.body
  br label %cond.end

cond.false:                                       ; preds = %while.body
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %5, %cond.true ], [ %6, %cond.false ]
  br label %while.cond9

while.cond9:                                      ; preds = %while.body15, %cond.end
  %C1_segend.0 = phi i32 [ %kC.0, %cond.end ], [ %inc, %while.body15 ]
  %cmp10 = icmp slt i32 %C1_segend.0, %3
  br i1 %cmp10, label %land.rhs11, label %land.end14

land.rhs11:                                       ; preds = %while.cond9
  %arrayidx12 = getelementptr inbounds i32, i32* %C1_crd, i32 %C1_segend.0
  %7 = load i32, i32* %arrayidx12, align 4
  %cmp13 = icmp eq i32 %7, %cond
  br label %land.end14

land.end14:                                       ; preds = %land.rhs11, %while.cond9
  %8 = phi i1 [ false, %while.cond9 ], [ %cmp13, %land.rhs11 ]
  br i1 %8, label %while.body15, label %while.end

while.body15:                                     ; preds = %land.end14
  %inc = add nsw i32 %C1_segend.0, 1
  br label %while.cond9

while.end:                                        ; preds = %land.end14
  %cmp16 = icmp eq i32 %5, %cond
  br i1 %cmp16, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %while.end
  %cmp17 = icmp eq i32 %6, %cond
  br i1 %cmp17, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  br label %for.cond18

for.cond18:                                       ; preds = %for.inc, %if.then
  %jC.0 = phi i32 [ %kC.0, %if.then ], [ %inc29, %for.inc ]
  %cmp19 = icmp slt i32 %jC.0, %C1_segend.0
  br i1 %cmp19, label %for.body20, label %for.end

for.body20:                                       ; preds = %for.cond18
  %arrayidx21 = getelementptr inbounds i32, i32* %C2_crd, i32 %jC.0
  %9 = load i32, i32* %arrayidx21, align 4
  %mul = mul nsw i32 %i.0, %A2_dimension
  %add22 = add nsw i32 %mul, %9
  %arrayidx23 = getelementptr inbounds double, double* %A_vals, i32 %add22
  %10 = load double, double* %arrayidx23, align 4
  %arrayidx24 = getelementptr inbounds double, double* %B_vals, i32 %kB.0
  %11 = load double, double* %arrayidx24, align 4
  %arrayidx25 = getelementptr inbounds double, double* %C_vals, i32 %jC.0
  %12 = load double, double* %arrayidx25, align 4
  %mul26 = fmul double %11, %12
  %add27 = fadd double %10, %mul26
  %arrayidx28 = getelementptr inbounds double, double* %A_vals, i32 %add22
  store double %add27, double* %arrayidx28, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body20
  %inc29 = add nsw i32 %jC.0, 1
  br label %for.cond18

for.end:                                          ; preds = %for.cond18
  br label %if.end

if.end:                                           ; preds = %for.end, %land.lhs.true, %while.end
  %cmp30 = icmp eq i32 %5, %cond
  %conv = zext i1 %cmp30 to i32
  %add31 = add nsw i32 %kB.0, %conv
  %add32 = add nsw i32 %kC.0, %C1_segend.0
  br label %while.cond

while.end33:                                      ; preds = %land.end
  br label %for.inc34

for.inc34:                                        ; preds = %while.end33
  %inc35 = add nsw i32 %i.0, 1
  br label %for.cond

for.end36:                                        ; preds = %for.cond
  ret void
}

attributes #0 = { noinline nounwind ssp "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+armv4t,+soft-float,+strict-align,-crypto,-d32,-dotprod,-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fp16,-fp16fml,-fp64,-fpregs,-fullfp16,-mve,-mve.fp,-neon,-thumb-mode,-vfp2,-vfp2sp,-vfp3,-vfp3d16,-vfp3d16sp,-vfp3sp,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp" "unsafe-fp-math"="false" "use-soft-float"="true" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{!"clang version 10.0.0 (git@github.com:kromancer/llvm-project.git d32170dbd5b0d54436537b6b75beaf44324e0c28)"}
