import argparse
from os import chdir, getcwd, makedirs
from contextlib import contextmanager

from spmv import make_build_dir_and_cd_to_it, apply_passes

from mlir import ir
from mlir.dialects import func
from mlir.dialects.linalg.opdsl import lang as dsl
from mlir.dialects import sparse_tensor as st

no_parallelization = ["sparse-reinterpret-map",
                      "sparsification{parallelization-strategy=none}",
                      "sparse-tensor-codegen",
                      "func-bufferize",
                      "bufferization-bufferize",
                      "convert-scf-to-cf",
                      "convert-to-llvm"]

vectorized = ["sparse-reinterpret-map",
              "sparsification{parallelization-strategy=none}",
              "sparse-vectorization{vl=16}",
              "sparse-tensor-codegen",
              "func-bufferize",
              "bufferization-bufferize",
              "convert-scf-to-cf",
              "convert-to-llvm"]

omp = ["sparse-reinterpret-map",
       "sparsification{parallelization-strategy=any-storage-any-loop}",
       "sparse-tensor-codegen",
       "func-bufferize",
       "bufferization-bufferize",
       "convert-scf-to-openmp",
       "convert-to-llvm"]

pipelines = {"no_parallelization": no_parallelization, "vectorized": vectorized, "omp": omp}


@dsl.linalg_structured_op
def matmul_dsl(
    A=dsl.TensorDef(dsl.T, dsl.S.I, dsl.S.J),
    B=dsl.TensorDef(dsl.T, dsl.S.J),
    C=dsl.TensorDef(dsl.T, dsl.S.I, output=True),
):
    C[dsl.D.i] += A[dsl.D.i, dsl.D.j] * B[dsl.D.j]


def build_spmv(rows: int, cols: int, attr: st.EncodingAttr):
    module = ir.Module.create()
    f64 = ir.F64Type.get()
    a = ir.RankedTensorType.get([rows, cols], f64, attr)
    b = ir.RankedTensorType.get([cols], f64)
    c = ir.RankedTensorType.get([rows], f64)
    arguments = [a, b, c]
    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(*arguments)
        def spmv(*args):
            return matmul_dsl(args[0], args[1], outs=[args[2]])

    return module


@contextmanager
def make_and_switch_dir(dir):
    current_dir = getcwd()
    try:
        makedirs(dir)
        chdir(dir)
        yield
    finally:
        chdir(current_dir)


def get_args():
    parser = argparse.ArgumentParser(description="Process rows and cols.")
    parser.add_argument("-r", "--rows", type=int, default=1024, help="Number of rows (default=1024)")
    parser.add_argument("-c", "--cols", type=int, default=1024, help="Number of columns (default=1024)")

    args = parser.parse_args()
    return args.rows, args.cols


def main():
    rows, cols = get_args()
    make_build_dir_and_cd_to_it(__file__)

    with ir.Context() as ctx, ir.Location.unknown():

        level = [st.DimLevelType.dense, st.DimLevelType.compressed]
        ordering = ir.AffineMap.get_permutation([0, 1])
        bitwidth = 0

        attr = st.EncodingAttr.get(level, ordering, ordering, bitwidth, bitwidth)
        module = build_spmv(rows, cols, attr)
        func = str(module.operation.regions[0].blocks[0].operations[0].operation)
        module = ir.Module.parse(func)

        with open("spmv.mlir", "w") as f:
            f.write(str(module))

        for name, passes in pipelines.items():
            with make_and_switch_dir(name):
                apply_passes(str(module), passes)


if __name__ == "__main__":
    main()
