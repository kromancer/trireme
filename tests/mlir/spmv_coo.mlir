// RUN: mlir-opt %s --sparsification=pd=32 --cse | FileCheck %s

//CHECK:          %[[pd:.+]] = arith.constant 32 : index
//CHECK:           %[[c:.+]] = bufferization.to_memref %arg1
//CHECK:           %[[a:.+]] = bufferization.to_memref %arg2
//CHECK:      %[[Bi_pos:.+]] = sparse_tensor.positions %arg0
//CHECK:      %[[Bi_crd:.+]] = sparse_tensor.coordinates %arg0 {level = 0 : index}
//CHECK: %[[Bi_crd_size:.+]] = memref.load %[[Bi_pos]][%c1]
//CHECK:      %[[Bj_crd:.+]] = sparse_tensor.coordinates %arg0 {level = 1 : index}

// prefetch for writes on a[Bi_crd[]]
//CHECK: scf.while (%[[seg_start:.+]] = {{.+}}, %[[seg_end:.+]] = {{.+}}) : (index, index) -> (index, index)
//CHECK:      %[[bound:.+]] = arith.subi %[[Bi_crd_size]], %c1
//CHECK:      arith.select
//CHECK:       %[[pref:.+]] = memref.load %[[Bi_crd]]
//CHECK:      memref.prefetch %[[a]][%[[pref]]], write, locality<2>, data

// prefetch for reads on c[Bj_crd[jj]]
//CHECK: scf.for %[[iB:.+]] = %[[seg_start]] to %[[seg_end]]
//CHECK:     %[[iB_plus_pd:.+]] = arith.addi %[[iB]], %[[pd]]
//CHECK:            %[[cmp:.+]] = arith.cmpi ult, %[[iB_plus_pd]], %[[bound]]
//CHECK:            %[[sel:.+]] = arith.select %[[cmp]], %[[iB_plus_pd]], %[[bound]]
//CHECK:           %[[pref:.+]] = memref.load %[[Bj_crd]][%[[sel]]]
//CHECK:     memref.prefetch %[[c]][%[[pref]]], read, locality<2>, data

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
