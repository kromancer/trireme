import tensorflow as tf


def SpMV(sparse_matrix, dense_vector):
    return tf.sparse.sparse_dense_matmul(sparse_matrix, dense_vector)

# Define input signatures
input_signature = [
    tf.SparseTensorSpec(shape=(3, 4), dtype=tf.float64),
    tf.TensorSpec(shape=(4, 1), dtype=tf.float64)  # Assuming the 2-D shape for dense_vector
]

# Get concrete function
concrete_function = tf.function(SpMV).get_concrete_function(*input_signature)

# Convert to MLIR
with open("spmv.tf.mlir", "w") as f:
    f.write(tf.mlir.experimental.convert_function(concrete_function))
