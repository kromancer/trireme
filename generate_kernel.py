import argparse
from contextlib import contextmanager
from platform import machine
from os import chdir, environ, getcwd, makedirs
from pathlib import Path
from subprocess import run
from typing import Optional, Tuple

import numpy as np

from mlir import ir
from mlir.dialects import func
from mlir.dialects import sparse_tensor as st
from mlir.passmanager import *

from argument_parsers import add_dimension_args
from common import SparseFormats, make_work_dir_and_cd_to_it
from kernels import spmv_dsl, spvv_dsl, spmm_dsl

pipelines = {
    "no-opt":
    ["sparse-assembler{direct-out=true}",
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
     "convert-scf-to-cf",
     "expand-strided-metadata",
     "finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false}",
     "convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false}",
     "convert-openmp-to-llvm",
     "reconcile-unrealized-casts"]
}

# defer execution by using lambdas, requires an active MLIR "Context"
np_to_mlir_type = {
    np.dtype("float64"): lambda: ir.F64Type.get(),
    np.dtype("int32"): lambda: ir.IntegerType.get_signless(32),
    np.dtype("int64"): lambda: ir.IntegerType.get_signless(64)
}


def apply_passes(src: str, kernel: str, pipeline: str, main_fun: Optional[str] = None,
                 index_type: np.dtype = np.dtype("int64")) -> Tuple[ir.Module, str]:
    out_file_name: str

    def run_pass(mlir_opt_pass: str):
        nonlocal module

        # Adapt the width of the index type
        if "index-bitwidth" in mlir_opt_pass:
            mlir_opt_pass = mlir_opt_pass.replace("index-bitwidth=0",
                                                  f"index-bitwidth={np_to_mlir_type[index_type]().width}")

        run_pass.call_count += 1
        try:
            pm = PassManager.parse(f"builtin.module({mlir_opt_pass})")
        except ValueError:
            pm = PassManager.parse(f"builtin.module(func.func({mlir_opt_pass}))")

        try:
            pm.run(module.operation)
        except Exception:
            print(f"Failure in: {kernel}.{run_pass.call_count}.{mlir_opt_pass}")
            raise

        out = f"{kernel}.{run_pass.call_count}.{mlir_opt_pass}.mlir"

        # Inject main after the "sparse-assembler" pass
        if mlir_opt_pass.startswith("sparse-assembler") and main_fun is not None:
            ops = "".join([str(o.operation) for o in module.operation.regions[0].blocks[0].operations])
            module = ir.Module.parse(ops + main_fun)

        with open(out, "w") as f:
            f.write(str(module))
        return out

    module = ir.Module.parse(src)
    run_pass.call_count = 0
    out_file_name = ""
    for p in pipelines[pipeline]:
        try:
            out_file_name = run_pass(p)
        except Exception:
            print(f"Pipeline: {pipeline}")
            raise

    return module, out_file_name


def get_compressed_vec_encoding() -> st.EncodingAttr:
    return st.EncodingAttr.parse(
        "#sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>")


def get_csr_encoding() -> st.EncodingAttr:
    return st.EncodingAttr.parse(
        "#sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>")


def get_coo_encoding() -> st.EncodingAttr:
    return st.EncodingAttr.parse(
        "#sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton(soa)) }>")


get_encoding = {SparseFormats.CSR: get_csr_encoding,
                SparseFormats.COO: get_coo_encoding}


def make_spvv_mlir_module(size: int, t: np.dtype = np.dtype("float64")) -> ir.Module:
    module = ir.Module.create()
    t = np_to_mlir_type[t]()
    a = ir.RankedTensorType.get([], t)
    b = ir.RankedTensorType.get([size], t, get_compressed_vec_encoding())
    c = ir.RankedTensorType.get([size], t, get_compressed_vec_encoding())
    arguments = [a, b, c]
    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(*arguments)
        def spvv(*args):
            return spvv_dsl(args[1], args[2], outs=[args[0]])

    return module


def make_spmv_mlir_module(rows: int, cols: int, enc: SparseFormats, t: np.dtype = np.dtype("float64"),
                          is_sparse_vec: bool = False) -> ir.Module:
    module = ir.Module.create()
    t = np_to_mlir_type[t]()
    a = ir.RankedTensorType.get([rows], t)
    B = ir.RankedTensorType.get([rows, cols], t, get_encoding[enc]())
    if is_sparse_vec:
        c = ir.RankedTensorType.get([cols], t, get_compressed_vec_encoding())
    else:
        c = ir.RankedTensorType.get([cols], t)
    arguments = [a, B, c]
    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(*arguments)
        def spmv(*args):
            return spmv_dsl(args[1], args[2], outs=[args[0]])

    return module


def make_spmm_mlir_module(res_rows: int, res_cols: int, inner_dim: int, enc_first: SparseFormats,
                          enc_other: SparseFormats, t: np.dtype = np.dtype("float64"),
                          dense_out: bool = False) -> ir.Module:
    module = ir.Module.create()
    t = np_to_mlir_type[t]()
    A = ir.RankedTensorType.get([res_rows, res_cols], t) \
        if dense_out else (
        ir.RankedTensorType.get([res_rows, res_cols], t, get_csr_encoding()))
    B = ir.RankedTensorType.get([res_rows, inner_dim], t, get_encoding[enc_first]())
    C = ir.RankedTensorType.get([inner_dim, res_cols], t, get_encoding[enc_other]())
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


def generate_spvv(size: int):
    with ir.Context() as ctx, ir.Location.unknown():
        module = make_spvv_mlir_module(size)
        generate(module, f"spvv", translate_to_llvm_ir=True)


def generate_spmv(rows: int, cols: int, enc: SparseFormats, is_sparse_vec: bool):
    with ir.Context() as ctx, ir.Location.unknown():
        module = make_spmv_mlir_module(rows, cols, enc, is_sparse_vec=is_sparse_vec)
        generate(module, f"spmv_{enc}" + ("_spvec" if is_sparse_vec else ""), translate_to_llvm_ir=True)


def generate_spmm(res_rows: int, res_cols: int, inner_dim: int, enc_first: SparseFormats, enc_other: SparseFormats):
    with ir.Context() as ctx, ir.Location.unknown():
        module = make_spmm_mlir_module(res_rows, res_cols, inner_dim, enc_first, enc_other)
        generate(module, f"spmm_{enc_first}_{enc_other}", translate_to_llvm_ir=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mlir, from the linalg to the llvm dialect, for given kernel")
    subparsers = parser.add_subparsers(dest="kernel", help="Which kernel to generate")

    spvv_parser = argparse.ArgumentParser(add_help=False)
    add_dimension_args(spvv_parser, 1)
    subparsers.add_parser("spvv", help="Sparse-Vector X Sparse-Vector (SpVV)",
                          parents=[spvv_parser])

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

    if args.kernel == "spvv":
        generate_spvv(args.i)
    if args.kernel == "spmv":
        generate_spmv(args.i, args.j, SparseFormats.CSR, is_sparse_vec=False)
        generate_spmv(args.i, args.j, SparseFormats.COO, is_sparse_vec=False)
        generate_spmv(args.i, args.j, SparseFormats.CSR, is_sparse_vec=True)
        generate_spmv(args.i, args.j, SparseFormats.COO, is_sparse_vec=True)
    elif args.kernel == "spmm":
        generate_spmm(args.i, args.j, args.k, SparseFormats.CSR, SparseFormats.CSR)
        generate_spmm(args.i, args.j, args.k, SparseFormats.COO, SparseFormats.COO)


if __name__ == "__main__":
    main()
