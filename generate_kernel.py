import argparse
from contextlib import contextmanager
from platform import machine
from os import chdir, environ, getcwd, makedirs
from pathlib import Path
from subprocess import run
from typing import Optional, Tuple

from mlir import ir
from mlir.dialects import func
from mlir.dialects.linalg.opdsl import lang as dsl
from mlir.dialects import sparse_tensor as st
from mlir.passmanager import *

from argument_parsers import add_dimension_args
from common import Encodings, make_work_dir_and_cd_to_it

pipelines = {
    "no-opt":
    ["sparse-assembler",
     "sparse-reinterpret-map",
     "sparsification{enable-runtime-library=false}",
     "sparse-tensor-codegen",
     "func-bufferize",
     "reconcile-unrealized-casts",
     "sparse-storage-specifier-to-llvm",
     "canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}",
     "finalizing-bufferize",
     "convert-scf-to-cf",
     "expand-strided-metadata",
     "finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false}",
     "convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false}",
     "reconcile-unrealized-casts"],

    "pref":
    ["sparse-assembler",
     "sparse-reinterpret-map",
     "sparsification{enable-runtime-library=false enable-prefetches=true}",
     "sparse-tensor-codegen",
     "func-bufferize",
     "reconcile-unrealized-casts",
     "sparse-storage-specifier-to-llvm",
     "canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}",
     "finalizing-bufferize",
     "convert-scf-to-cf",
     "expand-strided-metadata",
     "finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false}",
     "convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false}",
     "reconcile-unrealized-casts"],

    "vect-vl4":
    ["sparse-assembler",
     "sparse-reinterpret-map",
     "sparsification{enable-runtime-library=false}",
     "sparse-vectorization{vl=4}",
     "sparse-tensor-codegen",
     "func-bufferize",
     "reconcile-unrealized-casts",
     "sparse-storage-specifier-to-llvm",
     "canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}",
     "finalizing-bufferize",
     "convert-scf-to-cf",
     "expand-strided-metadata",
     "lower-affine",
     "finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false}",
     f"convert-vector-to-llvm{{{'enable-x86vector' if machine() == 'x86_64' else 'enable-arm-neon'}}}",
     "convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false}",
     "reconcile-unrealized-casts"],

    "omp":
    ["sparse-assembler",
     "sparse-reinterpret-map",
     "sparsification{enable-runtime-library=false parallelization-strategy=any-storage-any-loop}",
     "sparse-tensor-codegen",
     "func-bufferize",
     "reconcile-unrealized-casts",
     "sparse-storage-specifier-to-llvm",
     "canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}",
     "finalizing-bufferize",
     "convert-scf-to-openmp",
     "expand-strided-metadata",
     "finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false}",
     "convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false}",
     "convert-openmp-to-llvm",
     "reconcile-unrealized-casts"]
}


def apply_passes(src: str, kernel: str, pipeline: str, main: Optional[str] = None) -> Tuple[ir.Module, str]:
    out_file_name: str

    def run_pass(mlir_opt_pass: str):
        nonlocal module

        run_pass.call_count += 1
        try:
            pm = PassManager.parse(f"builtin.module({mlir_opt_pass})")
        except ValueError:
            pm = PassManager.parse(f"builtin.module(func.func({mlir_opt_pass}))")

        pm.run(module.operation)
        out = f"{kernel}.{run_pass.call_count}.{mlir_opt_pass}.mlir"

        # Inject main after the "sparse-assembler" pass
        if mlir_opt_pass == "sparse-assembler" and main is not None:
            kernel_func = str(module.operation.regions[0].blocks[0].operations[0].operation)
            kernel_func_internal = str(module.operation.regions[0].blocks[0].operations[1].operation)
            module = ir.Module.parse(kernel_func + kernel_func_internal + main)

        with open(out, "w") as f:
            f.write(str(module))
        return out

    module = ir.Module.parse(src)
    run_pass.call_count = 0
    out_file_name = ""
    for p in pipelines[pipeline]:
        out_file_name = run_pass(p)

    return module, out_file_name


def get_csr_encoding() -> st.EncodingAttr:
    return st.EncodingAttr.parse(
        "#sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>")


def get_coo_encoding() -> st.EncodingAttr:
    return st.EncodingAttr.parse(
        "#sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton(soa)) }>")


get_encoding = {Encodings.CSR: get_csr_encoding,
                Encodings.COO: get_coo_encoding}


@dsl.linalg_structured_op
def spmv_dsl(
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
            return spmv_dsl(args[0], args[1], outs=[args[2]])

    return module


@dsl.linalg_structured_op
def spmm_dsl(
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
            return spmm_dsl(args[1], args[2], outs=[args[0]])

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


def generate(module: ir.Module, kernel_name: str, translate_to_llvm_ir: bool = False):
    with make_and_switch_dir(kernel_name):
        with open(f"{kernel_name}.mlir", "w") as f:
            f.write(str(module))

        for p in pipelines:
            with make_and_switch_dir(p):
                _, last_output = apply_passes(str(module), kernel_name, p)

            if translate_to_llvm_ir:
                mlir_translate = Path(environ['LLVM_PATH']) / "bin/mlir-translate"
                assert mlir_translate.exists()

                translate_cmd = [str(mlir_translate), "--mlir-to-llvmir", Path(p) / last_output, "-o", f"{kernel_name}_{p}.ll"]
                run(translate_cmd, check=True)


def generate_spmv(rows: int, cols: int, enc: Encodings):
    with ir.Context() as ctx, ir.Location.unknown():
        module = make_spmv_mlir_module(rows, cols, enc)
        generate(module, "spmv", translate_to_llvm_ir=True)


def generate_spmm(res_rows: int, res_cols: int, inner_dim: int, enc_first: Encodings, enc_other: Encodings):
    with ir.Context() as ctx, ir.Location.unknown():
        module = make_spmm_mlir_module(res_rows, res_cols, inner_dim, enc_first, enc_other)
        generate(module, f"spmm_{enc_first}_{enc_other}", translate_to_llvm_ir=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mlir, from the linalg to the llvm dialect, for given kernel")
    subparsers = parser.add_subparsers(dest="kernel", help="Which kernel to generate")

    spmv_parser = argparse.ArgumentParser(add_help=False)
    add_dimension_args(spmv_parser, 2)
    subparsers.add_parser("spmv", help="Sparse-Matrix X Dense-Vector (SpMV)",
                          parents=[spmv_parser])

    spmm_parser = argparse.ArgumentParser(add_help=False)
    add_dimension_args(spmm_parser, 3)
    subparsers.add_parser("spmm", help="Sparse-Matrix X Sparse-Matrix (SpMM)",
                          parents=[spmm_parser])

    return parser.parse_args()


def main():
    args = parse_args()
    make_work_dir_and_cd_to_it(__file__)

    if args.kernel == "spmv":
        generate_spmv(args.i, args.j, Encodings.CSR)
    elif args.kernel == "spmm":
        generate_spmm(args.i, args.j, args.k, Encodings.CSR, Encodings.CSR)
        generate_spmm(args.i, args.j, args.k, Encodings.COO, Encodings.COO)


if __name__ == "__main__":
    main()
