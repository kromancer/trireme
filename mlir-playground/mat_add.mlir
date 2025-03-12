


#CSR = #sparse_tensor.encoding<{ map = (i, j) -> (i : dense, j : compressed) }>
#CSC = #sparse_tensor.encoding<{ map = (i, j) -> (i : dense, j : compressed) }>


//#m1 = affine_map<(i, j) -> (i, j)>
//#m2 = affine_map<(i, j) -> (i, j)>
//#m3 = affine_map<(i, j) -> (i, j)>

//#trait = {
//  indexing_maps = [#m1, #m2, #m3],
//  iterator_types = ["parallel", "parallel"],
//  sorted = true
//}

module {
  func.func @mat_add(%A: tensor<6x6xf64>,
                     %B: tensor<6x6xf64, #CSR>,
                     %C: tensor<6x6xf64, #CSC>) -> tensor<6x6xf64>
  {
//    %res = linalg.generic #trait
//           ins(%B, %C: tensor<6x6xf64, #CSR>, tensor<6x6xf64, #CSC>)
//           outs(%A: tensor<6x6xf64>) {
//             ^bb0(%b: f64, %c: f64, %a: f64):
//               %0 = arith.addf %b, %c: f64
//               linalg.yield %0: f64
//           } -> tensor<6x6xf64>
    %res1 = linalg.add ins(%A, %B: tensor<6x6xf64>, tensor<6x6xf64, #CSR>) outs(%A: tensor<6x6xf64>) -> tensor<6x6xf64>
    %res2 = linalg.add ins(%res1, %C: tensor<6x6xf64>, tensor<6x6xf64, #CSC>) outs(%A: tensor<6x6xf64>) -> tensor<6x6xf64>
    return %res2: tensor<6x6xf64>
  }
}
