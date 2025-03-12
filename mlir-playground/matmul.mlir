
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

module {
  func.func @spmm(%lhs: tensor<512x512xf32, #sparse>,
                  %rhs: tensor<512x512xf32, #sparse>,
                  %output: tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
  {
    %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32, #sparse>, tensor<512x512xf32, #sparse>) outs(%output: tensor<512x512xf32, #sparse>) -> tensor<512x512xf32, #sparse>
    return %matmul: tensor<512x512xf32, #sparse>
  }
}
