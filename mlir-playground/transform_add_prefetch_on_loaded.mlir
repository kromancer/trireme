//RUN: mlir-opt %s --pass-pipeline="builtin.module(transform-interpreter)" | FileCheck %s

module {
 func.func private @foo(%bound: index, %Bj_crd: memref<?xindex>, %vec: memref<?xf32>) {
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c1024 = arith.constant 1024 : index
   %pd = arith.constant 42 : index

   scf.for %jj = %c0 to %c1024 step %c1 {
      %j = memref.load %Bj_crd[%jj] : memref<?xindex>
      %jj_plus_pd = arith.addi %jj, %pd : index
      %sel = arith.cmpi ult, %jj_plus_pd, %bound : index
      %load_idx = arith.select %sel, %jj_plus_pd, %bound : index
      %pref_idx = memref.load %Bj_crd[%load_idx] : memref<?xindex>
      memref.prefetch %vec[%pref_idx], read, locality<2>, data : memref<?xf32>
   }
   return
 }
} // module

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_prefetches(%prefetch: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %prefetch ["memref.prefetch"] : !transform.any_op
    transform.yield %prefetch : !transform.any_op
  }

  transform.named_sequence @set_locality_hint(%prefetch: !transform.op<"memref.prefetch"> {transform.readonly}) {
    transform.memref.set_prefetch_locality_hint %prefetch 1 : !transform.op<"memref.prefetch">
    transform.yield
  }

  transform.named_sequence @add_prefetch_of_operand(%prefetch: !transform.op<"memref.prefetch"> {transform.readonly}) {
     %operand = transform.get_operand %prefetch[1] : (!transform.op<"memref.prefetch">) -> !transform.any_value
     %load_op = transform.get_defining_op %operand : (!transform.any_value) -> !transform.op<"memref.load">
     transform.print %load_op: !transform.op<"memref.load">
     transform.yield
  }

  transform.named_sequence @__transform_main(%ir: !transform.any_op) {
      %changed_loc_hint = transform.foreach_match in %ir
          @match_prefetches -> @set_locality_hint
        : (!transform.any_op) -> (!transform.any_op)

      transform.foreach_match in %changed_loc_hint
          @match_prefetches -> @add_prefetch_of_operand
        : (!transform.any_op) -> (!transform.any_op)

      transform.yield
  }
}
