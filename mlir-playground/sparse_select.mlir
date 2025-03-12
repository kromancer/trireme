#CSR = #sparse_tensor.encoding<{
    map = (i, j) -> (i: dense, j: compressed)
}>

#map = affine_map<(i,j) -> (i,j)>

#trait = {
  indexing_maps = [#map, #map],
  iterator_types = ["parallel", "parallel"],
  sorted = true
}

module {
  func.func @spmv(%B: tensor<10x10xf64, #CSR>) -> tensor<10x10xf64, #CSR>
  {
    %A = tensor.empty(): tensor<10x10xf64, #CSR>
    %3 = linalg.generic #trait
         ins (%B: tensor<10x10xf64, #CSR>)
         outs(%A: tensor<10x10xf64, #CSR>) {

         ^bb0(%bij: f64, %aij: f64) :
             %i = linalg.index 0: index
             %j = linalg.index 1: index

             %sel = sparse_tensor.select %bij : f64 {
             ^bb0(%arg0: f64):
                 %keep = arith.cmpi eq, %i, %j : index
                 sparse_tensor.yield %keep : i1
             }

             linalg.yield %sel : f64
         } -> tensor<10x10xf64, #CSR>

     return %3 : tensor<10x10xf64, #CSR>
  }
}

#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

module {
  func.func @spmv(%arg0: tensor<10x10xf64, #sparse>) -> tensor<10x10xf64, #sparse> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %0 = tensor.empty() : tensor<10x10xf64, #sparse>
    %1 = sparse_tensor.values %arg0 : tensor<10x10xf64, #sparse> to memref<?xf64>
    %2 = sparse_tensor.positions %arg0 {level = 1 : index} : tensor<10x10xf64, #sparse> to memref<?xindex>
    %3 = sparse_tensor.coordinates %arg0 {level = 1 : index} : tensor<10x10xf64, #sparse> to memref<?xindex>
    %4 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %0) -> (tensor<10x10xf64, #sparse>) {
      %6 = memref.load %2[%arg1] : memref<?xindex>
      %7 = arith.addi %arg1, %c1 : index
      %8 = memref.load %2[%7] : memref<?xindex>
      %9 = scf.for %arg3 = %6 to %8 step %c1 iter_args(%arg4 = %arg2) -> (tensor<10x10xf64, #sparse>) {
        %10 = memref.load %3[%arg3] : memref<?xindex>
        %11 = memref.load %1[%arg3] : memref<?xf64>
        %12 = arith.cmpi eq, %arg1, %10 : index
        %13 = scf.if %12 -> (tensor<10x10xf64, #sparse>) {
          %inserted = tensor.insert %11 into %arg4[%arg1, %10] : tensor<10x10xf64, #sparse>
          scf.yield %inserted : tensor<10x10xf64, #sparse>
        } else {
          scf.yield %arg4 : tensor<10x10xf64, #sparse>
        }
        scf.yield %13 : tensor<10x10xf64, #sparse>
      } {"Emitted from" = "linalg.generic"}
      scf.yield %9 : tensor<10x10xf64, #sparse>
    } {"Emitted from" = "linalg.generic"}
    %5 = sparse_tensor.load %4 hasInserts : tensor<10x10xf64, #sparse>
    return %5 : tensor<10x10xf64, #sparse>
  }
}
