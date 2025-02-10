// RUN: mlir-opt %s --sparsification=pd=32 --cse | FileCheck %s

// CHECK:          %[[pd:.+]] = arith.constant 32
// CHECK:           %[[c:.+]] = bufferization.to_memref %arg1
// CHECK:      %[[Bi_pos:.+]] = sparse_tensor.positions %arg0 {level = 0 : index}
// CHECK:      %[[Bi_crd:.+]] = sparse_tensor.coordinates %arg0 {level = 0 : index}
// CHECK: %[[Bi_crd_size:.+]] = memref.load %[[Bi_pos]][%c1]
// CHECK:      %[[Bj_pos:.+]] = sparse_tensor.positions %arg0 {level = 1 : index}
// CHECK:      %[[Bj_crd:.+]] = sparse_tensor.coordinates %arg0 {level = 1 : index}
// CHECK: %[[Bj_crd_size:.+]] = memref.load %[[Bj_pos]][%[[Bi_crd_size]]]

// For all non-empty rows
// CHECK: scf.for %[[ii:.+]] = {{.+}} to %[[Bi_crd_size]] step %c1
//            For all non-zeroes in row
// CHECK:     {{.+}} = scf.for %[[jj:.+]] = %[[B2_pos_i:.+]] to {{.+}} step %c1 iter_args({{.+}} = {{.+}}) -> (f64) {
// CHECK:              %[[bound:.+]] = arith.subi %[[Bj_crd_size]], %c1
// CHECK:         %[[jj_plus_pd:.+]] = arith.addi %[[jj]], %[[pd]]
// CHECK:                %[[cmp:.+]] = arith.cmpi ult, %[[jj_plus_pd]], %[[bound]]
// CHECK:                %[[sel:.+]] = arith.select %[[cmp]], %[[jj_plus_pd]], %[[bound]]
// CHECK:               %[[pref:.+]] = memref.load %[[Bj_crd]][%[[sel]]]
// CHECK:         memref.prefetch %[[c]][%[[pref]]], read, locality<2>, data

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>
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
