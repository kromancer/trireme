// RUN: mlir-opt %s --sparsification=enable-prefetches --cse | FileCheck %s

//CHECK-DAG: %[[a:.+]] = bufferization.to_memref %arg2
//CHECK-DAG: %[[c:.+]] = bufferization.to_memref %arg1
//CHECK-DAG: %[[Bi_crd:.+]] = sparse_tensor.coordinates %arg0 {level = 0 : index}
//CHECK-DAG: %[[Bj_crd:.+]] = sparse_tensor.coordinates %arg0 {level = 1 : index}
//CHECK-DAG: %[[Bi_pos:.+]] = sparse_tensor.positions %arg0
//CHECK-DAG: %[[Bi_pos_1:.+]] = memref.load %[[Bi_pos]][%c1]

// prefetch for writes on a[Bi_crd[]]
//CHECK:      scf.while (%[[seg_start:.+]] = {{.+}}, %[[seg_end:.+]] = {{.+}})
//CHECK:      %[[pref:.+]] = memref.load %[[Bi_crd]][%[[seg_end]]]
//CHECK-NEXT: memref.prefetch %[[a]][%[[pref]]], write, locality<3>, data

// prefetch for reads on c[Bj_crd[iB]]
//CHECK:      scf.for %[[iB:.+]] = %[[seg_start]] to %[[seg_end]]
//CHECK:      %[[iB_plus_2dist:.+]] = arith.addi %[[iB]]
//CHECK-NEXT: memref.prefetch %[[Bj_crd]][%[[iB_plus_2dist]]], read, locality<0>, data
//CHECK-NEXT: %[[iB_plus_dist:.+]] = arith.addi %[[iB]]
//CHECK-NEXT: %[[cmp:.+]] = arith.cmpi ult, %[[iB_plus_dist]], %[[Bi_pos_1]]
//CHECK-NEXT: %[[sel:.+]] = arith.select %[[cmp]], %[[iB_plus_dist]], %[[Bi_pos_1]]
//CHECK-NEXT: %[[pref:.+]] = memref.load %[[Bj_crd]][%[[sel]]]
//CHECK-NEXT: memref.prefetch %[[c]][%[[pref]]], read, locality<3>, data

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton(soa)) }>
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
