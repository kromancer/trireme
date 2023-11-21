



func.func @spMxM(%arg0: tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton) }>>,
                 %arg1: tensor<4x2xf64>, %arg2: tensor<3x2xf64>) -> tensor<3x2xf64>
{

  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0)>,
                                        affine_map<(d0, d1, d2) -> (d0, d2)>,
					affine_map<(d0, d1, d2) -> (d1, d2)>],
					iterator_types = ["reduction", "parallel", "parallel"]
		       }
		       ins(%arg0, %arg1 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton) }>>,
		                          tensor<4x2xf64>)
		       outs(%arg2 : tensor<3x2xf64>)
  {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %1 = arith.mulf %in, %in_0 : f64
      %2 = arith.addf %out, %1 : f64
      linalg.yield %2 : f64
  } -> tensor<3x2xf64>
  
  return %0 : tensor<3x2xf64>
}







