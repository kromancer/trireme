import argparse
from enum import Enum
from contextlib import contextmanager
from platform import machine
from os import chdir, getcwd, makedirs
from typing import List

from mlir import ir
from mlir.dialects import func
from mlir.dialects.linalg.opdsl import lang as dsl
from mlir.dialects import sparse_tensor as st
from mlir.passmanager import *

from common import Encodings, get_spmm_arg_parser, get_spmv_arg_parser, make_work_dir_and_cd_to_it

pipelines = {
    "no-opt":
    ["sparsification-and-bufferization{sparse-emit-strategy=functional}",
     "sparse-storage-specifier-to-llvm",
     "convert-scf-to-cf",
     "expand-strided-metadata",
     "finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false}",
     "convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false}",
     "reconcile-unrealized-casts"],

    "vect-vl4":
    ["sparsification-and-bufferization{sparse-emit-strategy=functional vl=4}",
     "sparse-storage-specifier-to-llvm",
     "convert-scf-to-cf",
     "expand-strided-metadata",
     "lower-affine",
     "finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false}",
     f"convert-vector-to-llvm{{{'enable-x86vector' if machine() == 'x86_64' else 'enable-arm-neon'}}}",
     "convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false}",
     "reconcile-unrealized-casts"],

    "omp":
    ["sparse-reinterpret-map",
     "sparsification{enable-runtime-library=false parallelization-strategy=any-storage-any-loop}",
     "sparse-tensor-codegen",
     "func-bufferize",
     "finalizing-bufferize",
     "sparse-storage-specifier-to-llvm",
     "convert-scf-to-openmp",
     "expand-strided-metadata",
     "finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false}",
     "convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false}",
     "convert-openmp-to-llvm",
     "reconcile-unrealized-casts"]
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

    module = ir.Module.parse(src)
    run_pass.call_count = 0
    for p in pipelines[pipeline]:
        run_pass(p)

    return module


def get_csr_encoding() -> st.EncodingAttr:
    builder = st.EncodingAttr.build_level_type
    fmt = st.LevelFormat
    csr = [builder(fmt.dense), builder(fmt.compressed)]
    ordering = ir.AffineMap.get_permutation([0, 1])
    bitwidth = 0
    return st.EncodingAttr.get(csr, ordering, ordering, bitwidth, bitwidth)


def get_coo_encoding() -> st.EncodingAttr:
    builder = st.EncodingAttr.build_level_type
    fmt = st.LevelFormat
    prop = st.LevelProperty
    coo = [builder(fmt.compressed, [prop.non_unique]), builder(fmt.singleton)]
    ordering = ir.AffineMap.get_permutation([0, 1])
    bitwidth = 0
    return st.EncodingAttr.get(coo, ordering, ordering, bitwidth, bitwidth)


get_encoding = {Encodings.CSR: get_csr_encoding,
                Encodings.COO: get_coo_encoding}


@dsl.linalg_structured_op
def mat_vec_dsl(
    A=dsl.TensorDef(dsl.T, dsl.S.I, dsl.S.J),
    B=dsl.TensorDef(dsl.T, dsl.S.J),
    C=dsl.TensorDef(dsl.T, dsl.S.I, output=True)
):
    C[dsl.D.i] += A[dsl.D.i, dsl.D.j] * B[dsl.D.j]


def make_spmv_mlir_module(rows: int, cols: int, enc: Encodings) -> ir.Module:
    module = ir.Module.create()
    f64 = ir.F64Type.get()
    a = ir.RankedTensorType.get([rows, cols], f64, get_encoding[enc]())
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


def make_spmm_mlir_module(res_rows: int, res_cols: int, inner_dim: int, enc_first: Encodings, enc_other: Encodings) -> ir.Module:
    module = ir.Module.create()
    f64 = ir.F64Type.get()
    A = ir.RankedTensorType.get([res_rows, res_cols], f64)
    B = ir.RankedTensorType.get([res_rows, inner_dim], f64, get_encoding[enc_first]())
    C = ir.RankedTensorType.get([inner_dim, res_cols], f64, get_encoding[enc_other]())
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


def generate(module: ir.Module, kernel_name: str):
    with make_and_switch_dir(kernel_name):
        with open(f"{kernel_name}.mlir", "w") as f:
            f.write(str(module))

        for p in pipelines:
            with make_and_switch_dir(p):
                apply_passes(str(module), kernel_name, p)


def generate_spmv(rows: int, cols: int, enc: Encodings):
    with ir.Context() as ctx, ir.Location.unknown():
        module = make_spmv_mlir_module(rows, cols, enc)
        generate(module, "spmv")


def generate_spmm(res_rows: int, res_cols: int, inner_dim: int, enc_first: Encodings, enc_other: Encodings):
    with ir.Context() as ctx, ir.Location.unknown():
        module = make_spmm_mlir_module(res_rows, res_cols, inner_dim, enc_first, enc_other)
        generate(module, f"spmm_{enc_first}_{enc_other}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mlir, from the linalg to the llvm dialect, for given kernel")

    subparsers = parser.add_subparsers(dest="kernel", help="Which kernel to generate")
    subparsers.add_parser("spmv", help="Sparse-Matrix X Dense-Vector (SpMV)",
                          parents=[get_spmv_arg_parser(with_pd=False, with_loc_hint=False, with_density=False)])
    subparsers.add_parser("spmm", help="Sparse-Matrix X Sparse-Matrix (SpMM)",
                          parents=[get_spmm_arg_parser(with_density=False)])

    return parser.parse_args()


def main():
    args = parse_args()
    make_work_dir_and_cd_to_it(__file__)

    if args.kernel == "spmv":
        generate_spmv(args.rows, args.cols, Encodings.CSR)
    elif args.kernel == "spmm":
        generate_spmm(args.rows, args.cols, args.inner_dim_size, Encodings.CSR, Encodings.CSR)
        generate_spmm(args.rows, args.cols, args.inner_dim_size, Encodings.COO, Encodings.COO)


if __name__ == "__main__":
    main()
