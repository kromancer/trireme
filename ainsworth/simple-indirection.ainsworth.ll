; ModuleID = 'simple-indirection.ll'
source_filename = "simple-indirection.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx10.19.0"

; Function Attrs: noinline nounwind ssp
define void @compute(i32* noalias nocapture readonly %key_buff1, i32* noalias nocapture readonly %key_buff2, i32 %num_keys) local_unnamed_addr #0 {
entry:
  %cmp3 = icmp sgt i32 %num_keys, 0
  br i1 %cmp3, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %loop_bound = add i32 %num_keys, -1
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.04 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]

  ; compute address of key_buff2[i]
  %addr_of_key_buff2_i = getelementptr inbounds i32, i32* %key_buff2, i32 %i.04

  ; compute the address of key_buff2[i + 64] (2 * distance)
  %i_plus_2d = add nuw i32 %i.04, 64
  %2 = getelementptr inbounds i32, i32* %key_buff2, i32 %i_plus_2d
  %addr_key_buff2_prefetch = bitcast i32* %2 to i8*

  ; compute address of key_buff1[key_buff2[i]]
  %4 = load i32, i32* %addr_of_key_buff2_i, align 4
  %addr_key_buff1_i = getelementptr inbounds i32, i32* %key_buff1, i32 %4

  ; find min(loop_bound, i + distance)
  %5 = add nuw i32 %i.04, 32
  %6 = icmp slt i32 %loop_bound, %5
  %i_plus_d = select i1 %6, i32 %loop_bound, i32 %5

  ; load key_buff2[i + distance]
  %8 = getelementptr inbounds i32, i32* %key_buff2, i32 %i_plus_d
  %9 = load i32, i32* %8, align 4

  ; compute address of key_buff1[key_buff2[i + distance]]
  %10 = getelementptr inbounds i32, i32* %key_buff1, i32 %9
  %addr_key_buff1_prefetch = bitcast i32* %10 to i8*

  ; call some_function(key_buff1[key_buff2[i]])
  %12 = load i32, i32* %addr_key_buff1_i, align 4
  tail call void @some_function(i32 %12) #3
  
  %inc = add nuw nsw i32 %i.04, 1
  %exitcond = icmp eq i32 %inc, %num_keys

  ; Prefetches
  call void @llvm.prefetch.p0i8(i8* nonnull %addr_key_buff2_prefetch, i32 0, i32 3, i32 1)
  call void @llvm.prefetch.p0i8(i8*         %addr_key_buff1_prefetch, i32 0, i32 3, i32 1)
  
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare void @some_function(i32) local_unnamed_addr #1

; Function Attrs: inaccessiblemem_or_argmemonly nounwind willreturn
declare void @llvm.prefetch.p0i8(i8* nocapture readonly, i32 immarg, i32 immarg, i32) #2

attributes #0 = { noinline nounwind ssp "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+armv4t,+soft-float,+strict-align,-crypto,-d32,-dotprod,-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fp16,-fp16fml,-fp64,-fpregs,-fullfp16,-mve,-mve.fp,-neon,-thumb-mode,-vfp2,-vfp2sp,-vfp3,-vfp3d16,-vfp3d16sp,-vfp3sp,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+armv4t,+soft-float,+strict-align,-crypto,-d32,-dotprod,-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fp16,-fp16fml,-fp64,-fpregs,-fullfp16,-mve,-mve.fp,-neon,-thumb-mode,-vfp2,-vfp2sp,-vfp3,-vfp3d16,-vfp3d16sp,-vfp3sp,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #2 = { inaccessiblemem_or_argmemonly nounwind willreturn }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{!"clang version 10.0.0 (git@github.com:kromancer/llvm-project.git d32170dbd5b0d54436537b6b75beaf44324e0c28)"}
