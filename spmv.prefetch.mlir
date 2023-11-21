module {

  func.func @spMV(%sparseT: tensor<1024x1024xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>,
                  %vec: tensor<1024xf64>,
		  %res: tensor<1024xf64>) -> tensor<1024xf64> {
		  
    %c3 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    %positions_buff = sparse_tensor.positions %sparseT {level = 1 : index} : tensor<1024x1024xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xindex>
    %coordinates_buff = sparse_tensor.coordinates %sparseT {level = 1 : index} : tensor<1024x1024xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xindex>
    %values_buff = sparse_tensor.values %sparseT : tensor<1024x1024xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xf64>
    
    %vec_buff = bufferization.to_memref %vec : memref<1024xf64>
    %res_buff = bufferization.to_memref %res : memref<1024xf64>
    
    scf.for %i = %c0 to %c3 step %c1 {
      %6 = memref.load %res_buff[%i] : memref<1024xf64>
      
      %offset_start = memref.load %positions_buff[%i] : memref<?xindex>
      
      %i_plus_one = arith.addi %i, %c1 : index
      %offset_end = memref.load %positions_buff[%i_plus_one] : memref<?xindex>
      
      %res_i = scf.for %j = %offset_start to %offset_end step %c1 iter_args(%arg5 = %6) -> (f64) {
      
        %col_idx = memref.load %coordinates_buff[%j] : memref<?xindex>
        %t_val = memref.load %values_buff[%j] : memref<?xf64>

        // vec_buff[coordinates[positions[i]]] aka C[B[A[i]]]
        %v_val = memref.load %vec_buff[%col_idx] : memref<1024xf64>
	
        %14 = arith.mulf %t_val, %v_val : f64
        %15 = arith.addf %arg5, %14 : f64
        scf.yield %15 : f64	
      }
      
      memref.store %res_i, %res_buff[%i] : memref<1024xf64>
    } 
    
    %5 = bufferization.to_tensor %res_buff restrict : memref<1024xf64>
    return %5 : tensor<1024xf64>
  }
  
}