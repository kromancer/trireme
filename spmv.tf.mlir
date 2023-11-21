module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1575 : i32}}
{

  func.func @__inference_SpMV_8(%arg0: tensor<?x2xi64> {tf._user_specified_name = "sparse_matrix"},
                                %arg1: tensor<?xf64>   {tf._user_specified_name = "sparse_matrix"},
				%arg2: tensor<2xi64>   {tf._user_specified_name = "sparse_matrix"},
				%arg3: tensor<4x1xf64> {tf._user_specified_name = "dense_vector"}) -> tensor<?x1xf64>
				attributes {
				allow_soft_placement = false,
				tf.entry_function = {control_outputs = "",
				                     inputs = "sparse_matrix, sparse_matrix_1, sparse_matrix_2, dense_vector",
						     outputs = "identity_RetVal"}}
  {
    %0 = "tf.SparseTensorDenseMatMul"(%arg0, %arg1, %arg2, %arg3) {adjoint_a = false, adjoint_b = false, device = ""} : (tensor<?x2xi64>, tensor<?xf64>, tensor<2xi64>, tensor<4x1xf64>) -> tensor<?x1xf64>
    %1 = "tf.Identity"(%0) {device = ""} : (tensor<?x1xf64>) -> tensor<?x1xf64>
    return %1 : tensor<?x1xf64>
  }
  
}
