module {
  func.func @spMV(%arg0: tensor<3x4xf64,
                  #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>,
  	          %arg1: tensor<4xf64>,
		  %arg2: tensor<3xf64>) -> tensor<3xf64>
  {

    %0 = linalg.generic {
       	   indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
	   		    affine_map<(d0, d1) -> (d0)>,
			    affine_map<(d0, d1) -> (d1)>],
			    iterator_types = ["reduction", "parallel"] }
			    
	  ins(%arg0, %arg1 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, tensor<4xf64>)
	  outs(%arg2 : tensor<3xf64>)
	  {
	    ^bb0(%in: f64, %in_0: f64, %out: f64):
               %1 = arith.mulf %in, %in_0 : f64
               %2 = arith.addf %out, %1 : f64
               linalg.yield %2 : f64
	  } -> tensor<3xf64>
    
    return %0 : tensor<3xf64>
  }

  func.func @main(%arg0: tensor<3x4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> attributes {llvm.emit_c_interface} {
    %0 = sparse_tensor.convert %arg0 : tensor<3x4xf64> to tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
    %1 = call @spMV(%0, %arg1, %arg2) : (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, tensor<4xf64>, tensor<3xf64>) -> tensor<3xf64>
    return %1 : tensor<3xf64>
  }
}

