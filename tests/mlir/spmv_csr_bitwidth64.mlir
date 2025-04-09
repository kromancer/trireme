// RUN: mlir-opt %s --sparsification=pd=32 | FileCheck %s

// CHECK:          %[[pd2:.+]] = arith.constant 64 : index
// CHECK:           %[[pd:.+]] = arith.constant 32 : index
// CHECK:            %[[c:.+]] = bufferization.to_memref %arg1
// CHECK:       %[[Bj_pos:.+]] = sparse_tensor.positions
// CHECK:       %[[Bj_crd:.+]] = sparse_tensor.coordinates
// CHECK:         %[[load:.+]] = memref.load %[[Bj_pos]][%c1024]
// CHECK:  %[[Bj_crd_size:.+]] = arith.index_cast %[[load]] : i64 to index

// Check inner loop for prefetching
// CHECK: {{.+}} = scf.for %[[jj:.+]] = {{.+}} to {{.+}} step %c1 iter_args({{.+}} = {{.+}}) -> (f64) {
// CHECK:          %[[bound:.+]] = arith.subi %[[Bj_crd_size]], %c1
// CHECK:    %[[jj_plus_pd2:.+]] = arith.addi %[[jj]], %[[pd2]]
// CHECK:           memref.prefetch %[[Bj_crd]][%[[jj_plus_pd2]]], read, locality<2>, data
// CHECK:     %[[jj_plus_pd:.+]] = arith.addi %[[jj]], %[[pd]]
// CHECK:            %[[cmp:.+]] = arith.cmpi ult, %[[jj_plus_pd]], %[[bound]]
// CHECK:            %[[sel:.+]] = arith.select %[[cmp]], %[[jj_plus_pd]], %[[bound]]
// CHECK:          %[[load2:.+]] = memref.load %[[Bj_crd]][%[[sel]]]
// CHECK:           %[[pref:.+]] = arith.index_cast %[[load2]] : i64 to index
// CHECK:           memref.prefetch %[[c]][%[[pref]]], read, locality<2>, data

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed), posWidth=64, crdWidth=64 }>
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
