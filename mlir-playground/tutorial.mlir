#out = affine_map<(i, j) -> (i, j)>
#a = affine_map<(i, j) -> (i, j)>
#b = affine_map<(i, j) -> (i, j)>


module {

  func.func private @foo(%out: memref<4x8xf32>, %a: memref<4x8xf32>, %b: memref<4x8xf32>, %thres: f32) {

    // 'linalg.generic' op result #0 must be variadic of ranked tensor of any type values, but got 'memref<4x8xf32>'
    linalg.generic {
       indexing_maps = [#a, #b, #out],
       iterator_types = ["parallel", "parallel"]
       }
       ins(%a, %b : memref<4x8xf32>, memref<4x8xf32>) outs(%out : memref<4x8xf32>) {
    ^bb0(%a_: f32, %b_: f32, %_: f32):
      %1 = arith.addf %a_, %b_ : f32
      %2 = arith.maximumf %1, %thres : f32
      linalg.yield %2 : f32
    }
    return
  }

}
