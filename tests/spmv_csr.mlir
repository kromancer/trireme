// RUN: mlir-opt %s --sparsification=enable-prefetches | FileCheck %s

// CHECK-DAG:  %[[C_vals:.+]] = bufferization.to_memref %arg1 : memref<1024xf64>
// CHECK-DAG:  %[[B2_pos:.+]] = sparse_tensor.positions %[[B:.+]]   {level = 1 : index} : tensor<1024x1024xf64, #sparse> to memref<?xindex>
// CHECK-DAG:  %[[B2_crd:.+]] = sparse_tensor.coordinates %[[B:.+]] {level = 1 : index} : tensor<1024x1024xf64, #sparse> to memref<?xindex>

// Check inner loop for prefetching
// CHECK:       %[[UPPERB:.+]] = memref.load %[[B2_pos]][%c1024] : memref<?xindex>
// CHECK:       {{.+}} = scf.for %[[jB:.+]] = %[[B2_pos_i:.+]] to {{.*}} step %c1 iter_args({{.+}} = {{.+}}) -> (f64) {
// CHECK-DAG:     %[[jB_plus_64:.+]] = arith.addi %[[jB]], %c64 : index
// CHECK:         memref.prefetch %[[B2_crd]][%[[jB_plus_64]]], read, locality<0>, data : memref<?xindex>
// CHECK:         %[[jB_plus_32:.+]] = arith.addi %[[jB]], %c32 : index
// CHECK:         %[[CMP:.+]] = arith.cmpi ult, %[[jB_plus_32]], %[[UPPERB]] : index
// CHECK:         %[[SEL:.+]] = arith.select %[[CMP]], %[[jB_plus_32]], %[[UPPERB]] : index
// CHECK:         %[[PREF:.+]] = memref.load %[[B2_crd]][%[[SEL]]] : memref<?xindex>
// CHECK:         memref.prefetch %[[C_vals]][%[[PREF]]], read, locality<0>, data : memref<1024xf64>

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
