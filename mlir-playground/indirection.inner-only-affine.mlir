module {

func.func @reduce(%pos: memref<?xindex>, %vals: memref<?xf64>, %res: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %rows = memref.dim %pos, %c0 : memref<?xindex>

    %f0 = arith.constant 0.0 : f64
    %c1 = arith.constant 1 : index
    affine.for %i = %c0 to %rows {

        %j_start = affine.load %pos[%i]: memref<?xindex>
        %j_end = affine.load %pos[%i + 1] : memref<?xindex>

        %r_sum = func.call @reduce_inner(%j_start, %j_end, %vals) : (index, index, memref<?xf64>) -> f64
        affine.store %r_sum, %res[%i] : memref<?xf64>
    }

    return
}

func.func @reduce_inner(%j_start: index, %j_end: index, %vals: memref<?xf64>) -> f64 {

    %f0 = arith.constant 0.0 : f64
    %sum = affine.for %j = %j_start to %j_end iter_args(%acc = %f0) -> f64 {
        %t = affine.load %vals[%j] : memref<?xf64>
        %row_sum = arith.addf %acc, %t : f64
        affine.yield %row_sum : f64
    }

    return %sum: f64
}


}
