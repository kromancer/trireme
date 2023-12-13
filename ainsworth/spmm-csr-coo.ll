; ModuleID = 'spmm-csr-coo.c'
source_filename = "spmm-csr-coo.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.19.0"

; Function Attrs: noinline nounwind ssp
define void @compute(i32 %B1_dimension, i32 %A2_dimension, double* noalias %A_vals, i32* noalias %B2_pos, i32* noalias %B2_crd, double* noalias %B_vals, i32* noalias %C1_pos, i32* noalias %C1_crd, i32* noalias %C2_crd, double* noalias %C_vals) #0 {
entry:
  %B1_dimension.addr = alloca i32, align 4
  %A2_dimension.addr = alloca i32, align 4
  %A_vals.addr = alloca double*, align 4
  %B2_pos.addr = alloca i32*, align 4
  %B2_crd.addr = alloca i32*, align 4
  %B_vals.addr = alloca double*, align 4
  %C1_pos.addr = alloca i32*, align 4
  %C1_crd.addr = alloca i32*, align 4
  %C2_crd.addr = alloca i32*, align 4
  %C_vals.addr = alloca double*, align 4
  %i = alloca i32, align 4
  %kB = alloca i32, align 4
  %pB2_end = alloca i32, align 4
  %kC = alloca i32, align 4
  %pC1_end = alloca i32, align 4
  %kB0 = alloca i32, align 4
  %kC0 = alloca i32, align 4
  %k = alloca i32, align 4
  %C1_segend = alloca i32, align 4
  %jC = alloca i32, align 4
  %j = alloca i32, align 4
  %jA = alloca i32, align 4
  store i32 %B1_dimension, i32* %B1_dimension.addr, align 4
  store i32 %A2_dimension, i32* %A2_dimension.addr, align 4
  store double* %A_vals, double** %A_vals.addr, align 4
  store i32* %B2_pos, i32** %B2_pos.addr, align 4
  store i32* %B2_crd, i32** %B2_crd.addr, align 4
  store double* %B_vals, double** %B_vals.addr, align 4
  store i32* %C1_pos, i32** %C1_pos.addr, align 4
  store i32* %C1_crd, i32** %C1_crd.addr, align 4
  store i32* %C2_crd, i32** %C2_crd.addr, align 4
  store double* %C_vals, double** %C_vals.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc34, %entry
  %0 = load i32, i32* %i, align 4
  %1 = load i32, i32* %B1_dimension.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end36

for.body:                                         ; preds = %for.cond
  %2 = load i32*, i32** %B2_pos.addr, align 4
  %3 = load i32, i32* %i, align 4
  %arrayidx = getelementptr inbounds i32, i32* %2, i32 %3
  %4 = load i32, i32* %arrayidx, align 4
  store i32 %4, i32* %kB, align 4
  %5 = load i32*, i32** %B2_pos.addr, align 4
  %6 = load i32, i32* %i, align 4
  %add = add nsw i32 %6, 1
  %arrayidx1 = getelementptr inbounds i32, i32* %5, i32 %add
  %7 = load i32, i32* %arrayidx1, align 4
  store i32 %7, i32* %pB2_end, align 4
  %8 = load i32*, i32** %C1_pos.addr, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %8, i32 0
  %9 = load i32, i32* %arrayidx2, align 4
  store i32 %9, i32* %kC, align 4
  %10 = load i32*, i32** %C1_pos.addr, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %10, i32 1
  %11 = load i32, i32* %arrayidx3, align 4
  store i32 %11, i32* %pC1_end, align 4
  br label %while.cond

while.cond:                                       ; preds = %if.end, %for.body
  %12 = load i32, i32* %kB, align 4
  %13 = load i32, i32* %pB2_end, align 4
  %cmp4 = icmp slt i32 %12, %13
  br i1 %cmp4, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %while.cond
  %14 = load i32, i32* %kC, align 4
  %15 = load i32, i32* %pC1_end, align 4
  %cmp5 = icmp slt i32 %14, %15
  br label %land.end

land.end:                                         ; preds = %land.rhs, %while.cond
  %16 = phi i1 [ false, %while.cond ], [ %cmp5, %land.rhs ]
  br i1 %16, label %while.body, label %while.end33

while.body:                                       ; preds = %land.end
  %17 = load i32*, i32** %B2_crd.addr, align 4
  %18 = load i32, i32* %kB, align 4
  %arrayidx6 = getelementptr inbounds i32, i32* %17, i32 %18
  %19 = load i32, i32* %arrayidx6, align 4
  store i32 %19, i32* %kB0, align 4
  %20 = load i32*, i32** %C1_crd.addr, align 4
  %21 = load i32, i32* %kC, align 4
  %arrayidx7 = getelementptr inbounds i32, i32* %20, i32 %21
  %22 = load i32, i32* %arrayidx7, align 4
  store i32 %22, i32* %kC0, align 4
  %23 = load i32, i32* %kB0, align 4
  %24 = load i32, i32* %kC0, align 4
  %cmp8 = icmp slt i32 %23, %24
  br i1 %cmp8, label %cond.true, label %cond.false

cond.true:                                        ; preds = %while.body
  %25 = load i32, i32* %kB0, align 4
  br label %cond.end

cond.false:                                       ; preds = %while.body
  %26 = load i32, i32* %kC0, align 4
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %25, %cond.true ], [ %26, %cond.false ]
  store i32 %cond, i32* %k, align 4
  %27 = load i32, i32* %kC, align 4
  store i32 %27, i32* %C1_segend, align 4
  br label %while.cond9

while.cond9:                                      ; preds = %while.body15, %cond.end
  %28 = load i32, i32* %C1_segend, align 4
  %29 = load i32, i32* %pC1_end, align 4
  %cmp10 = icmp slt i32 %28, %29
  br i1 %cmp10, label %land.rhs11, label %land.end14

land.rhs11:                                       ; preds = %while.cond9
  %30 = load i32*, i32** %C1_crd.addr, align 4
  %31 = load i32, i32* %C1_segend, align 4
  %arrayidx12 = getelementptr inbounds i32, i32* %30, i32 %31
  %32 = load i32, i32* %arrayidx12, align 4
  %33 = load i32, i32* %k, align 4
  %cmp13 = icmp eq i32 %32, %33
  br label %land.end14

land.end14:                                       ; preds = %land.rhs11, %while.cond9
  %34 = phi i1 [ false, %while.cond9 ], [ %cmp13, %land.rhs11 ]
  br i1 %34, label %while.body15, label %while.end

while.body15:                                     ; preds = %land.end14
  %35 = load i32, i32* %C1_segend, align 4
  %inc = add nsw i32 %35, 1
  store i32 %inc, i32* %C1_segend, align 4
  br label %while.cond9

while.end:                                        ; preds = %land.end14
  %36 = load i32, i32* %kB0, align 4
  %37 = load i32, i32* %k, align 4
  %cmp16 = icmp eq i32 %36, %37
  br i1 %cmp16, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %while.end
  %38 = load i32, i32* %kC0, align 4
  %39 = load i32, i32* %k, align 4
  %cmp17 = icmp eq i32 %38, %39
  br i1 %cmp17, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %40 = load i32, i32* %kC, align 4
  store i32 %40, i32* %jC, align 4
  br label %for.cond18

for.cond18:                                       ; preds = %for.inc, %if.then
  %41 = load i32, i32* %jC, align 4
  %42 = load i32, i32* %C1_segend, align 4
  %cmp19 = icmp slt i32 %41, %42
  br i1 %cmp19, label %for.body20, label %for.end

for.body20:                                       ; preds = %for.cond18
  %43 = load i32*, i32** %C2_crd.addr, align 4
  %44 = load i32, i32* %jC, align 4
  %arrayidx21 = getelementptr inbounds i32, i32* %43, i32 %44
  %45 = load i32, i32* %arrayidx21, align 4
  store i32 %45, i32* %j, align 4
  %46 = load i32, i32* %i, align 4
  %47 = load i32, i32* %A2_dimension.addr, align 4
  %mul = mul nsw i32 %46, %47
  %48 = load i32, i32* %j, align 4
  %add22 = add nsw i32 %mul, %48
  store i32 %add22, i32* %jA, align 4
  %49 = load double*, double** %A_vals.addr, align 4
  %50 = load i32, i32* %jA, align 4
  %arrayidx23 = getelementptr inbounds double, double* %49, i32 %50
  %51 = load double, double* %arrayidx23, align 4
  %52 = load double*, double** %B_vals.addr, align 4
  %53 = load i32, i32* %kB, align 4
  %arrayidx24 = getelementptr inbounds double, double* %52, i32 %53
  %54 = load double, double* %arrayidx24, align 4
  %55 = load double*, double** %C_vals.addr, align 4
  %56 = load i32, i32* %jC, align 4
  %arrayidx25 = getelementptr inbounds double, double* %55, i32 %56
  %57 = load double, double* %arrayidx25, align 4
  %mul26 = fmul double %54, %57
  %add27 = fadd double %51, %mul26
  %58 = load double*, double** %A_vals.addr, align 4
  %59 = load i32, i32* %jA, align 4
  %arrayidx28 = getelementptr inbounds double, double* %58, i32 %59
  store double %add27, double* %arrayidx28, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body20
  %60 = load i32, i32* %jC, align 4
  %inc29 = add nsw i32 %60, 1
  store i32 %inc29, i32* %jC, align 4
  br label %for.cond18

for.end:                                          ; preds = %for.cond18
  br label %if.end

if.end:                                           ; preds = %for.end, %land.lhs.true, %while.end
  %61 = load i32, i32* %kB0, align 4
  %62 = load i32, i32* %k, align 4
  %cmp30 = icmp eq i32 %61, %62
  %conv = zext i1 %cmp30 to i32
  %63 = load i32, i32* %kB, align 4
  %add31 = add nsw i32 %63, %conv
  store i32 %add31, i32* %kB, align 4
  %64 = load i32, i32* %C1_segend, align 4
  %65 = load i32, i32* %kC, align 4
  %add32 = add nsw i32 %65, %64
  store i32 %add32, i32* %kC, align 4
  br label %while.cond

while.end33:                                      ; preds = %land.end
  br label %for.inc34

for.inc34:                                        ; preds = %while.end33
  %66 = load i32, i32* %i, align 4
  %inc35 = add nsw i32 %66, 1
  store i32 %inc35, i32* %i, align 4
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
