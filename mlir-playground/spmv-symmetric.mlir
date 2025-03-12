#CSR = #sparse_tensor.encoding<{
   map = (i, j) -> (i: dense, j: compressed)
}>

#mB = affine_map<(i, j) -> (i, j)>
#mc = affine_map<(i, j) -> (j)>
#ma = affine_map<(i, j) -> (i)>


#trait = {
  indexing_maps = [#mB, #mc, #ma],
  iterator_types = ["parallel", "reduction"],
  sorted = true
}

#traitT = {
  indexing_maps = [#mB, #ma, #mc], // Flipped c/a maps
  iterator_types = ["reduction", "parallel"],
  sorted = true
}

module {
  func.func @spmv(%a: tensor<10xf64>,
                  %B: tensor<10x10xf64, #CSR>,
                  %c: tensor<10xf64>) -> tensor<10xf64>
  {
     %U = linalg.generic #trait
          ins(%B, %c: tensor<10x10xf64, #CSR>, tensor<10xf64>)
          outs(%a: tensor<10xf64>) {
          ^bb0(%bij: f64, %cj: f64, %ai: f64) :
               %t1 = arith.mulf %bij, %cj : f64
               %t2 = arith.addf %t1, %ai: f64
               linalg.yield %t2: f64
          } -> tensor<10xf64>

      %c0f = arith.constant 0.0: f64
      %res = linalg.generic #traitT
         ins(%B, %c: tensor<10x10xf64, #CSR>, tensor<10xf64>)
         outs(%U: tensor<10xf64>) {
         ^bb0(%bij: f64, %cj: f64, %ai: f64) :
           %row = linalg.index 0 : index
           %col = linalg.index 1 : index
           %acc = sparse_tensor.reduce %bij, %cj, %c0f: f64 {
             ^bb0(%bij_: f64, %cj_: f64):
               %t1 = arith.mulf %bij_, %cj_ : f64
               %t2 = arith.addf %t1, %ai: f64
               %is_diag = arith.cmpi eq, %row, %col : index
               %sel = arith.select %is_diag, %ai, %t2: f64
               sparse_tensor.yield %sel: f64
           }
           linalg.yield %acc: f64
         } -> tensor<10xf64>

     return %res : tensor<10xf64>
  }
}
