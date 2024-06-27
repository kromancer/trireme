import argparse
from contextlib import contextmanager
from platform import machine
from os import chdir, getcwd, makedirs

from mlir import ir
from mlir.dialects import func
from mlir.dialects.linalg.opdsl import lang as dsl
from mlir.dialects import sparse_tensor as st
from mlir.passmanager import *

from common import make_work_dir_and_cd_to_it

pipelines = {
    "no-opt":
    ["sparse-reinterpret-map",
     "sparsification{parallelization-strategy=none}",
     "sparse-tensor-codegen",
     "sparse-storage-specifier-to-llvm",
     "func-bufferize",
     "convert-scf-to-cf",
     "convert-to-llvm"],

    "vect-vl4":
    ["sparse-reinterpret-map",
     "sparsification{parallelization-strategy=none}",
     "sparse-vectorization{vl=4}",
     "sparse-tensor-codegen",
     "func-bufferize",
     "convert-scf-to-cf",
     f"convert-vector-to-llvm{{{'enable-x86vector' if machine() == 'x86_64' else 'enable-arm-neon'}}}",
     "lower-affine",
     "convert-arith-to-llvm",
     "convert-to-llvm",
     "reconcile-unrealized-casts"],

    "omp":
    ["sparse-reinterpret-map",
     "sparsification{parallelization-strategy=any-storage-any-loop}",
     "sparse-tensor-codegen",
     "func-bufferize",
     "convert-scf-to-openmp",
     "convert-to-llvm"]
}


def apply_passes(src: str, kernel: str, pipeline: str) -> ir.Module:

    def run_pass(mlir_opt_pass: str):
        run_pass.call_count += 1
        try:
            pm = PassManager.parse(f"builtin.module({mlir_opt_pass})")
        except ValueError:
            pm = PassManager.parse(f"builtin.module(func.func({mlir_opt_pass}))")

        pm.run(module.operation)
        with open(f"{kernel}.{run_pass.call_count}.{mlir_opt_pass}.mlir", "w") as f:
            f.write(str(module))

    with ir.Context():
        module = ir.Module.parse(src)

        run_pass.call_count = 0
        for p in pipelines[pipeline]:
            run_pass(p)

    return module


@dsl.linalg_structured_op
def mat_vec_dsl(
    A=dsl.TensorDef(dsl.T, dsl.S.I, dsl.S.J),
    B=dsl.TensorDef(dsl.T, dsl.S.J),
    C=dsl.TensorDef(dsl.T, dsl.S.I, output=True)
):
    C[dsl.D.i] += A[dsl.D.i, dsl.D.j] * B[dsl.D.j]


def make_spmv_mlir_module(rows: int, cols: int, attr: st.EncodingAttr) -> ir.Module:
    module = ir.Module.create()
    f64 = ir.F64Type.get()
    a = ir.RankedTensorType.get([rows, cols], f64, attr)
    b = ir.RankedTensorType.get([cols], f64)
    c = ir.RankedTensorType.get([rows], f64)
    arguments = [a, b, c]
    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(*arguments)
        def spmv(*args):
            return mat_vec_dsl(args[0], args[1], outs=[args[2]])

    return module


@dsl.linalg_structured_op
def mat_mat_dsl(
    B=dsl.TensorDef(dsl.T, dsl.S.I, dsl.S.K),
    C=dsl.TensorDef(dsl.T, dsl.S.K, dsl.S.J),
    A=dsl.TensorDef(dsl.T, dsl.S.I, dsl.S.J, output=True)
):
    A[dsl.D.i, dsl.D.j] += B[dsl.D.i, dsl.D.k] * C[dsl.D.k, dsl.D.j]


def make_spmm_mlir_module(rows: int, cols: int, enc_first: st.EncodingAttr, enc_other: st.EncodingAttr) -> ir.Module:
    module = ir.Module.create()
    f64 = ir.F64Type.get()
    A = ir.RankedTensorType.get([rows, cols], f64)
    B = ir.RankedTensorType.get([rows, cols], f64, enc_first)
    C = ir.RankedTensorType.get([cols, cols], f64, enc_other)
    arguments = [A, B, C]
    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(*arguments)
        def spmm(*args):
            return mat_mat_dsl(args[1], args[2], outs=[args[0]])

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


def generate(module: ir.Module, kernel_name: str):

    with make_and_switch_dir(kernel_name):
        with open(f"{kernel_name}.mlir", "w") as f:
            f.write(str(module))

        for p in pipelines:
            with make_and_switch_dir(p):
                apply_passes(str(module), kernel_name, p)


def generate_spmv(rows: int, cols: int):

    with ir.Context() as ctx, ir.Location.unknown():
        builder = st.EncodingAttr.build_level_type
        fmt = st.LevelFormat
        level = [builder(fmt.dense), builder(fmt.compressed)]
        ordering = ir.AffineMap.get_permutation([0, 1])
        bitwidth = 0

        attr = st.EncodingAttr.get(level, ordering, ordering, bitwidth, bitwidth)
        module = make_spmv_mlir_module(rows, cols, attr)

        generate(module, "spmv")


def generate_spmm(rows: int, cols: int):

    with ir.Context() as ctx, ir.Location.unknown():
        builder = st.EncodingAttr.build_level_type
        fmt = st.LevelFormat
        prop = st.LevelProperty
        csr = [builder(fmt.dense), builder(fmt.compressed)]
        coo = [builder(fmt.compressed, [prop.non_unique]), builder(fmt.singleton)]

        ordering = ir.AffineMap.get_permutation([0, 1])
        bitwidth = 0

        encoding_csr = st.EncodingAttr.get(csr, ordering, ordering, bitwidth, bitwidth)
        module = make_spmm_mlir_module(rows, cols, encoding_csr, encoding_csr)
        generate(module, "spmm_csr_csr")

        encoding_coo = st.EncodingAttr.get(coo, ordering, ordering, bitwidth, bitwidth)
        module = make_spmm_mlir_module(rows, cols, encoding_csr, encoding_coo)
        generate(module, "spmm_csr_coo")


if __name__ == "__main__":
    rows, cols = get_args()
    make_work_dir_and_cd_to_it(__file__)

    generate_spmv(rows, cols)
    generate_spmm(rows, cols)
