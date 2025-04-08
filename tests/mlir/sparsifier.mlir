// RUN: mlir-opt %s --sparsifier="enable-runtime-library=false pd=45" | FileCheck %s

// CHECK: llvm.intr.prefetch


#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed), posWidth=32, crdWidth=32 }>
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
