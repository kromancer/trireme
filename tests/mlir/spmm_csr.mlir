// RUN: mlir-opt %s --sparsification=pd=45 --cse | FileCheck %s

// CHECK:            %[[C:.+]] = bufferization.to_memref %arg2
// CHECK:       %[[Bj_pos:.+]] = sparse_tensor.positions
// CHECK:       %[[Bj_crd:.+]] = sparse_tensor.coordinates
// CHECK:  %[[Bj_crd_size:.+]] = memref.load %[[Bj_pos]][%c1024]

// outer loop
// CHECK: scf.for %[[i:.+]] = %c0 to %c1024 step %c1
// CHECK: %[[seg_start:.+]] = memref.load %[[Bj_pos]][%[[i]]] : memref<?xindex>

// middle loop
// CHECK: scf.for %[[jj:.+]] = %[[seg_start]] to {{.+}} step %c1
// CHECK:          %[[bound:.+]] = arith.subi %[[Bj_crd_size]], %c1
// CHECK:      %[[jj_plus_1:.+]] = arith.addi %[[jj]], %c1
// CHECK:      memref.prefetch %[[Bj_crd]][%[[jj_plus_1]]], read, locality<2>, data
// CHECK:            %[[cmp:.+]] = arith.cmpi ult, %[[jj_plus_1]], %[[bound]]
// CHECK:            %[[sel:.+]] = arith.select %[[cmp]], %[[jj_plus_1]], %[[bound]]
// CHECK:           %[[pref:.+]] = memref.load %[[Bj_crd]][%[[sel]]]
// CHECK:           memref.prefetch %[[C]][%[[pref]], %c0], read, locality<2>, data

#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
module {
  func.func @spmm(%arg0: tensor<1024x2xf64>, %arg1: tensor<1024x1024xf64, #sparse>, %arg2: tensor<1024x2xf64>) -> tensor<1024x2xf64> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg1, %arg2 : tensor<1024x1024xf64, #sparse>, tensor<1024x2xf64>) outs(%arg0 : tensor<1024x2xf64>) attrs =  {sorted = true} {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
    } -> tensor<1024x2xf64>
    return %0 : tensor<1024x2xf64>
  }
}
