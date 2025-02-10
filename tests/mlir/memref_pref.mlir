//RUN: mlir-opt %s --pass-pipeline="builtin.module(transform-interpreter)" | FileCheck %s

module {
 func.func private @foo(%arg0: memref<?xf32>, %arg1: index) {
   //CHECK: memref.prefetch %arg0[%arg1], read, locality<1>, data : memref<?xf32>
   memref.prefetch %arg0[%arg1], read, locality<2>, data : memref<?xf32>
   return
 }
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["memref.prefetch"]} in %arg0 : (!transform.any_op) -> !transform.op<"memref.prefetch">
    transform.memref.set_prefetch_locality_hint %0 1 : !transform.op<"memref.prefetch">
    transform.yield
  }
}
