module {

func.func @foo(%A: memref<?xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %N = memref.dim %A, %c0 : memref<?xf64>

    %f0 = arith.constant 0.0 : f64
    %sum = affine.for %i = 0 to %N iter_args(%acc = %f0) -> f64 {

        %t = affine.load %A[%i]: memref<?xf64>
        %next = arith.addf %acc, %t: f64

        affine.yield %next: f64
    }

    return %sum : f64
}

}
