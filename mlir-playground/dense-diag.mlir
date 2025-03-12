// mlir-opt --one-shot-bufferize --convert-linalg-to-loops diag.mlir

#mB = affine_map<(i) -> (i, i)>
#mc = affine_map<(i) -> (i)>
#ma = affine_map<(i) -> (i)>

#trait = {
  indexing_maps = [#mB, #mc, #ma],
  iterator_types = ["parallel"]
}

module {
  func.func @diag(%a: tensor<10xf64>,
                  %B: tensor<10x10xf64>,
                  %c: tensor<10xf64>) -> tensor<10xf64>
  {
    %res = linalg.generic #trait
         ins(%B, %c: tensor<10x10xf64>, tensor<10xf64>)
         outs(%a: tensor<10xf64>) {
    ^bb0(%bij: f64, %cj: f64, %acc: f64) :
      %tmp = arith.mulf %bij, %cj : f64
      %new_acc = arith.addf %tmp, %acc : f64
      linalg.yield %new_acc : f64
    } -> tensor<10xf64>

    return %res : tensor<10xf64>
  }
}
