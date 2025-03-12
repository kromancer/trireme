#mB = affine_map<(i, j) -> (i, j)>
#mc = affine_map<(i, j) -> (i)>
#ma = affine_map<(i, j) -> (j)>

// CSC: This will not generate prefetches
//#sparse = #sparse_tensor.encoding<{
//   map = (i, j) -> (j: dense, i: compressed)
//}>

// CSR: This will generate prefetches
#sparse = #sparse_tensor.encoding<{
   map = (i, j) -> (i: dense, j: compressed)
}>

#trait = {
  indexing_maps = [#mB, #mc, #ma],
  iterator_types = ["reduction", "parallel"]
}

module {
  func.func @spmv(%a: tensor<10xf64>,
                  %B: tensor<10x10xf64, #sparse>,
                  %c: tensor<10xf64>) -> tensor<10xf64>
  {

    %c0f = arith.constant 0.0: f64
    %3 = linalg.generic #trait
         ins(%B, %c: tensor<10x10xf64, #sparse>, tensor<10xf64>)
         outs(%a: tensor<10xf64>) {
         ^bb0(%bij: f64, %ci: f64, %aj: f64) :
           %row = linalg.index 0 : index
           %col = linalg.index 1 : index
           %acc = sparse_tensor.reduce %bij, %aj, %c0f: f64 {
             ^bb0(%bij_: f64, %aj_: f64):
               %t1 = arith.mulf %bij_, %ci : f64
               %t2 = arith.addf %t1, %aj_: f64
               %is_diag = arith.cmpi eq, %row, %col : index
               %sel = arith.select %is_diag, %t2, %aj_: f64
               sparse_tensor.yield %sel: f64
           }
           linalg.yield %acc: f64
         } -> tensor<10xf64>

     return %3 : tensor<10xf64>
  }
}
