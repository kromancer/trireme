#CSR = #sparse_tensor.encoding<{ map = (i, j) -> (i: dense, j: compressed) }>

#m1 = affine_map<(i, j) -> (i, j)>
#m2 = affine_map<(i, j) -> (j)>
#m3 = affine_map<(i, j) -> (i)>

#trait = {
  indexing_maps = [#m1, #m2, #m3],
  iterator_types = ["parallel", "reduction"],
  sorted = true
}

module {
  func.func @spmv(%a: tensor<5xf64>,
                  %B: tensor<5x5xf64, #CSR>,
                  %c: tensor<5xf64>) -> tensor<5xf64>
  {
    %0 = linalg.generic #trait
         ins(%B, %c: tensor<5x5xf64, #CSR>, tensor<5xf64>)
         outs(%a: tensor<5xf64>) {
    ^bb0(%bij: f64, %cj: f64, %ai: f64) :
      %f0 = arith.constant 0.0 : f64
      %v = sparse_tensor.unary %bij : f64 to f64
        present={
          ^bb0(%arg0: f64):
            sparse_tensor.yield %cj: f64
        }
        absent={}
      %r = sparse_tensor.reduce %bij, %v, %f0 : f64 {
        ^bb0(%p: f64, %q: f64):
          %add = arith.addf %p, %q : f64
          sparse_tensor.yield %add : f64
      }
      linalg.yield %r : f64
    } -> tensor<5xf64>
    return %0 : tensor<5xf64>
  }
}
