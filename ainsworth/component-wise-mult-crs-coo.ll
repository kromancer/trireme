; ModuleID = 'component-wise-mult-crs-coo.c'
source_filename = "component-wise-mult-crs-coo.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.19.0"

; Function Attrs: noinline nounwind ssp
define void @compute(i32 %max_j, double* noalias %A_vals, i32* noalias %B_pos, i32* noalias %B_crd, double* noalias %B_vals, i32* noalias %C1_pos, i32* noalias %C1_crd, i32* noalias %C2_crd, double* noalias %C_vals) #0 {
entry:
  %max_j.addr = alloca i32, align 4
  %A_vals.addr = alloca double*, align 4
  %B_pos.addr = alloca i32*, align 4
  %B_crd.addr = alloca i32*, align 4
  %B_vals.addr = alloca double*, align 4
  %C1_pos.addr = alloca i32*, align 4
  %C1_crd.addr = alloca i32*, align 4
  %C2_crd.addr = alloca i32*, align 4
  %C_vals.addr = alloca double*, align 4
  %iC = alloca i32, align 4
  %i = alloca i32, align 4
  %jB = alloca i32, align 4
  %pB2_end = alloca i32, align 4
  %jC = alloca i32, align 4
  %pC2_end = alloca i32, align 4
  %jB0 = alloca i32, align 4
  %jC0 = alloca i32, align 4
  %j = alloca i32, align 4
  %jA = alloca i32, align 4
  store i32 %max_j, i32* %max_j.addr, align 4
  store double* %A_vals, double** %A_vals.addr, align 4
  store i32* %B_pos, i32** %B_pos.addr, align 4
  store i32* %B_crd, i32** %B_crd.addr, align 4
  store double* %B_vals, double** %B_vals.addr, align 4
  store i32* %C1_pos, i32** %C1_pos.addr, align 4
  store i32* %C1_crd, i32** %C1_crd.addr, align 4
  store i32* %C2_crd, i32** %C2_crd.addr, align 4
  store double* %C_vals, double** %C_vals.addr, align 4
  %0 = load i32*, i32** %C1_pos.addr, align 4
  %arrayidx = getelementptr inbounds i32, i32* %0, i32 0
  %1 = load i32, i32* %arrayidx, align 4
  store i32 %1, i32* %iC, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, i32* %iC, align 4
  %3 = load i32*, i32** %C1_pos.addr, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %3, i32 1
  %4 = load i32, i32* %arrayidx1, align 4
  %cmp = icmp slt i32 %2, %4
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %5 = load i32*, i32** %C1_crd.addr, align 4
  %6 = load i32, i32* %iC, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %5, i32 %6
  %7 = load i32, i32* %arrayidx2, align 4
  store i32 %7, i32* %i, align 4
  %8 = load i32*, i32** %B_pos.addr, align 4
  %9 = load i32, i32* %i, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %8, i32 %9
  %10 = load i32, i32* %arrayidx3, align 4
  store i32 %10, i32* %jB, align 4
  %11 = load i32*, i32** %B_pos.addr, align 4
  %12 = load i32, i32* %i, align 4
  %add = add nsw i32 %12, 1
  %arrayidx4 = getelementptr inbounds i32, i32* %11, i32 %add
  %13 = load i32, i32* %arrayidx4, align 4
  store i32 %13, i32* %pB2_end, align 4
  %14 = load i32, i32* %iC, align 4
  store i32 %14, i32* %jC, align 4
  %15 = load i32, i32* %iC, align 4
  %add5 = add nsw i32 %15, 1
  store i32 %add5, i32* %pC2_end, align 4
  br label %while.cond

while.cond:                                       ; preds = %if.end, %for.body
  %16 = load i32, i32* %jB, align 4
  %17 = load i32, i32* %pB2_end, align 4
  %cmp6 = icmp slt i32 %16, %17
  br i1 %cmp6, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %while.cond
  %18 = load i32, i32* %jC, align 4
  %19 = load i32, i32* %pC2_end, align 4
  %cmp7 = icmp slt i32 %18, %19
  br label %land.end

land.end:                                         ; preds = %land.rhs, %while.cond
  %20 = phi i1 [ false, %while.cond ], [ %cmp7, %land.rhs ]
  br i1 %20, label %while.body, label %while.end

while.body:                                       ; preds = %land.end
  %21 = load i32*, i32** %B_crd.addr, align 4
  %22 = load i32, i32* %jB, align 4
  %arrayidx8 = getelementptr inbounds i32, i32* %21, i32 %22
  %23 = load i32, i32* %arrayidx8, align 4
  store i32 %23, i32* %jB0, align 4
  %24 = load i32*, i32** %C2_crd.addr, align 4
  %25 = load i32, i32* %jC, align 4
  %arrayidx9 = getelementptr inbounds i32, i32* %24, i32 %25
  %26 = load i32, i32* %arrayidx9, align 4
  store i32 %26, i32* %jC0, align 4
  %27 = load i32, i32* %jB0, align 4
  %28 = load i32, i32* %jC0, align 4
  %cmp10 = icmp slt i32 %27, %28
  br i1 %cmp10, label %cond.true, label %cond.false

cond.true:                                        ; preds = %while.body
  %29 = load i32, i32* %jB0, align 4
  br label %cond.end

cond.false:                                       ; preds = %while.body
  %30 = load i32, i32* %jC0, align 4
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %29, %cond.true ], [ %30, %cond.false ]
  store i32 %cond, i32* %j, align 4
  %31 = load i32, i32* %jB0, align 4
  %32 = load i32, i32* %j, align 4
  %cmp11 = icmp eq i32 %31, %32
  br i1 %cmp11, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %cond.end
  %33 = load i32, i32* %jC0, align 4
  %34 = load i32, i32* %j, align 4
  %cmp12 = icmp eq i32 %33, %34
  br i1 %cmp12, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %35 = load i32, i32* %i, align 4
  %36 = load i32, i32* %max_j.addr, align 4
  %mul = mul nsw i32 %35, %36
  %37 = load i32, i32* %j, align 4
  %add13 = add nsw i32 %mul, %37
  store i32 %add13, i32* %jA, align 4
  %38 = load double*, double** %B_vals.addr, align 4
  %39 = load i32, i32* %jB, align 4
  %arrayidx14 = getelementptr inbounds double, double* %38, i32 %39
  %40 = load double, double* %arrayidx14, align 4
  %41 = load double*, double** %C_vals.addr, align 4
  %42 = load i32, i32* %jC, align 4
  %arrayidx15 = getelementptr inbounds double, double* %41, i32 %42
  %43 = load double, double* %arrayidx15, align 4
  %mul16 = fmul double %40, %43
  %44 = load double*, double** %A_vals.addr, align 4
  %45 = load i32, i32* %jA, align 4
  %arrayidx17 = getelementptr inbounds double, double* %44, i32 %45
  store double %mul16, double* %arrayidx17, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %cond.end
  %46 = load i32, i32* %jB0, align 4
  %47 = load i32, i32* %j, align 4
  %cmp18 = icmp eq i32 %46, %47
  %conv = zext i1 %cmp18 to i32
  %48 = load i32, i32* %jB, align 4
  %add19 = add nsw i32 %48, %conv
  store i32 %add19, i32* %jB, align 4
  %49 = load i32, i32* %jC0, align 4
  %50 = load i32, i32* %j, align 4
  %cmp20 = icmp eq i32 %49, %50
  %conv21 = zext i1 %cmp20 to i32
  %51 = load i32, i32* %jC, align 4
  %add22 = add nsw i32 %51, %conv21
  store i32 %add22, i32* %jC, align 4
  br label %while.cond

while.end:                                        ; preds = %land.end
  br label %for.inc

for.inc:                                          ; preds = %while.end
  %52 = load i32, i32* %iC, align 4
  %inc = add nsw i32 %52, 1
  store i32 %inc, i32* %iC, align 4
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
