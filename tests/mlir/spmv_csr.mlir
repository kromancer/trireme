// RUN: mlir-opt %s --sparsification=enable-prefetches | FileCheck %s

// CHECK-DAG:  %[[c:.+]] = bufferization.to_memref %arg1
// CHECK-DAG:  %[[B2_pos:.+]] = sparse_tensor.positions
// CHECK-DAG:  %[[B2_crd:.+]] = sparse_tensor.coordinates

// Check inner loop for prefetching
// CHECK:      %[[upper:.+]] = memref.load %[[B2_pos]][%c1024]
// CHECK:      {{.+}} = scf.for %[[jB:.+]] = %[[B2_pos_i:.+]] to {{.+}} step %c1 iter_args({{.+}} = {{.+}}) -> (f64) {
// CHECK:      %[[jB_plus_2dist:.+]] = arith.addi %[[jB]]
// CHECK-NEXT: memref.prefetch %[[B2_crd]][%[[jB_plus_2dist]]], read, locality<0>, data
// CHECK-NEXT: %[[jB_plus_dist:.+]] = arith.addi %[[jB]]
// CHECK-NEXT: %[[cmp:.+]] = arith.cmpi ult, %[[jB_plus_dist]], %[[upper]]
// CHECK-NEXT: %[[sel:.+]] = arith.select %[[cmp]], %[[jB_plus_dist]], %[[upper]]
// CHECK-NEXT: %[[pref:.+]] = memref.load %[[B2_crd]][%[[sel]]]
// CHECK-NEXT: memref.prefetch %[[c]][%[[pref]]], read, locality<3>, data

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
module {
  func.func private @spmv(%arg0: tensor<1024x1024xf64, #sparse>, %arg1: tensor<1024xf64>, %arg2: tensor<1024xf64>) -> tensor<1024xf64> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1024x1024xf64, #sparse>, tensor<1024xf64>) outs(%arg2 : tensor<1024xf64>) attrs =  {sorted = true} {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<1024xf64>
    return %0 : tensor<1024xf64>
  }
}
