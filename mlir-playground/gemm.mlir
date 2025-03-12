#A = affine_map<(i, j, k) -> (i, k)>
#B = affine_map<(i, j, k) -> (k, j)>
#C = affine_map<(i, j, k) -> (i, j)>

module {

  // Computes C[i,j] = A[i,k] * B[k,j]
  func.func private @gemm(%C: tensor<4x4xf32>, %A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {

    %0 = linalg.generic {
       indexing_maps = [#A, #B, #C],
       iterator_types = ["parallel", "parallel", "reduction"]}
       ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%C : tensor<4x4xf32>) {
    ^bb0(%a_: f32, %b_: f32, %c_: f32):
      %1 = arith.mulf %a_, %b_ : f32
      %2 = arith.addf %1, %c_ : f32
      linalg.yield %2 : f32
    } -> tensor<4x4xf32>
    return %0: tensor<4x4xf32>
  }

}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %mm = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %loop, %loop2 = transform.structured.tile_using_for %mm tile_sizes[2,2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.print %loop : !transform.any_op
    transform.yield
  }
}
