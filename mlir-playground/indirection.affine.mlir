#map_i_plus_one = affine_map<(d0) -> (d0 + 1)>

module {

func.func @reduce(%pos: memref<?xindex>, %vals: memref<?xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %rows = memref.dim %pos, %c0 : memref<?xindex>

    %f0 = arith.constant 0.0 : f64
    %c1 = arith.constant 0 : index
    %sum = affine.for %i = %c0 to %rows iter_args(%acc = %f0) -> f64 {

        %j_start = affine.load %pos[%i]: memref<?xindex>

        %i_plus_one = affine.apply #map_i_plus_one(%i)
        %j_end = affine.load %pos[%i_plus_one] : memref<?xindex>

        %r_sum = affine.for %j = %j_start to %j_end iter_args(%acc_row = %f0) -> f64 {
            %t = affine.load %vals[%j] : memref<?xf64>
            %row_sum = arith.addf %acc_row, %t : f64
            affine.yield %row_sum : f64
        }

        %rows_sum = arith.addf %acc, %r_sum: f64
        affine.yield %rows_sum: f64
    }

    return %sum: f64
}

}
