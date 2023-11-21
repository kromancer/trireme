// mlir-opt --lower-sparse-ops-to-foreach --lower-sparse-foreach-to-scf --sparsification --sparse-reinterpret-map spmv.mlir > spmv.sparsification.mlir

module {

  func.func @spMV(%sparseT: tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, %vec: tensor<4xf64>, %res: tensor<3xf64>) -> tensor<3xf64> {
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    %positions_buff = sparse_tensor.positions %sparseT {level = 1 : index} : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xindex>
    %coordinates_buff = sparse_tensor.coordinates %sparseT {level = 1 : index} : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xindex>
    %values_buff = sparse_tensor.values %sparseT : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>> to memref<?xf64>
    
    %vec_buff = bufferization.to_memref %vec : memref<4xf64>
    %res_buff = bufferization.to_memref %res : memref<3xf64>
    
    scf.for %i = %c0 to %c3 step %c1 {
      %6 = memref.load %res_buff[%i] : memref<3xf64>
      
      %offset_start = memref.load %positions_buff[%i] : memref<?xindex>
      
      %i_plus_one = arith.addi %i, %c1 : index
      %offset_end = memref.load %positions_buff[%i_plus_one] : memref<?xindex>
      
      %res_i = scf.for %j = %offset_start to %offset_end step %c1 iter_args(%arg5 = %6) -> (f64) {
      
        %col_idx = memref.load %coordinates_buff[%j] : memref<?xindex>
        %t_val = memref.load %values_buff[%j] : memref<?xf64>

        // vec_buff[coordinates[positions[i]]] aka C[B[A[i]]]
        %v_val = memref.load %vec_buff[%col_idx] : memref<4xf64>
	
        %14 = arith.mulf %t_val, %v_val : f64
        %15 = arith.addf %arg5, %14 : f64
        scf.yield %15 : f64	
      }
      
      memref.store %res_i, %res_buff[%i] : memref<3xf64>
    } 
    
    %5 = bufferization.to_tensor %res_buff : memref<3xf64>
    return %5 : tensor<3xf64>
  }

  func.func @main(%arg0: tensor<3x4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> attributes {llvm.emit_c_interface} {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.alloc_tensor() : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
    %1 = bufferization.to_memref %arg0 : memref<3x4xf64>
    %2 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %0) -> (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>) {
      %5 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %arg4) -> (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>) {
        %6 = memref.load %1[%arg3, %arg5] : memref<3x4xf64>
        %7 = arith.cmpf une, %6, %cst : f64
        %8 = scf.if %7 -> (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>) {
          %inserted = sparse_tensor.insert %6 into %arg6[%arg3, %arg5] : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
          scf.yield %inserted : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
        } else {
          scf.yield %arg6 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
        }
        scf.yield %8 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
      } {"Emitted from" = "sparse_tensor.foreach"}
      scf.yield %5 : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
    } {"Emitted from" = "sparse_tensor.foreach"}
    
    %3 = sparse_tensor.load %2 hasInserts : tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>
    
    %4 = call @spMV(%3, %arg1, %arg2) : (tensor<3x4xf64, #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>>, tensor<4xf64>, tensor<3xf64>) -> tensor<3xf64>
    return %4 : tensor<3xf64>
  }
}

