import argparse
from contextlib import contextmanager
from platform import machine
from os import chdir, environ, getcwd, makedirs
from pathlib import Path
from subprocess import run
from typing import Optional, Tuple

import jinja2
import numpy as np

from mlir import ir
from mlir.dialects import func
from mlir.dialects import sparse_tensor as st
from mlir.passmanager import *

from argument_parsers import add_dimension_args, add_dtype_arg, add_opt_arg
from common import SparseFormats, make_work_dir_and_cd_to_it

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
     "convert-scf-to-openmp",
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
     "convert-cf-to-llvm",
     "convert-openmp-to-llvm",
     "canonicalize",
     "reconcile-unrealized-casts"]
}

# defer execution by using lambdas, requires an active MLIR "Context"
np_to_mlir_type = {
    np.dtype("float64"): lambda: ir.F64Type.get(),
    np.dtype("int32"): lambda: ir.IntegerType.get_signless(32),
    np.dtype("int64"): lambda: ir.IntegerType.get_signless(64),
    np.dtype("bool"): lambda: ir.IntegerType.get_signless(1)
}

to_mlir_type = {
    "float64": "f64",
    "float32": "f32",
    "int64": "i64",
    "int32": "i32",
    "bool": "i1"
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


encodings = {SparseFormats.CSR: "#sparse_tensor.encoding<{ map = (d0, d1) -> (d0: dense, d1: compressed) }>",
             SparseFormats.CSC: "#sparse_tensor.encoding<{ map = (d0, d1) -> (d1: dense, d0: compressed) }>",
             SparseFormats.COO: "#sparse_tensor.encoding<{ map = (d0, d1) -> (d0: compressed(nonunique), d1: singleton(soa)) }>"}


def get_jinja() -> jinja2.Environment:
    template_dir = Path(__file__).parent.resolve() / "templates"
    return jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))


def render_template_for_spmv(args: argparse.Namespace, encoding: str, is_sparse_vec: bool) -> str:
    jinja = get_jinja()
    if args.dtype == "bool":
        spmv_template = jinja.get_template("spmv_bool_semiring.mlir.jinja2")
    else:
        spmv_template = jinja.get_template("spmv_mult_semiring.mlir.jinja2")

    spmv_rendered = spmv_template.render(rows=args.i, cols=args.j, encoding=encoding, dtype=to_mlir_type[args.dtype],
                                         is_sparse_vec=is_sparse_vec)
    return spmv_rendered


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


def generate_spmv(args: argparse.Namespace, enc: SparseFormats, is_sparse_vec: bool):
    with ir.Context() as ctx, ir.Location.unknown():
        module = ir.Module.parse(render_template_for_spmv(args, encodings[enc], is_sparse_vec))
        generate(module, f"spmv_{enc}" + ("_spvec" if is_sparse_vec else ""), translate_to_llvm_ir=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mlir, from the linalg to the llvm dialect, for given kernel")
    add_opt_arg(parser)
    add_dtype_arg(parser)

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
        # generate_spvv(args.i)
        pass
    if args.kernel == "spmv":
        generate_spmv(args, SparseFormats.CSR, is_sparse_vec=False)
        generate_spmv(args, SparseFormats.COO, is_sparse_vec=False)
        generate_spmv(args, SparseFormats.CSC, is_sparse_vec=False)
        generate_spmv(args, SparseFormats.CSR, is_sparse_vec=True)
        generate_spmv(args, SparseFormats.COO, is_sparse_vec=True)
        generate_spmv(args, SparseFormats.CSC, is_sparse_vec=True)
    elif args.kernel == "spmm":
        # generate_spmm(args.i, args.j, args.k, SparseFormats.CSR, SparseFormats.CSR)
        # generate_spmm(args.i, args.j, args.k, SparseFormats.COO, SparseFormats.COO)
        pass


if __name__ == "__main__":
    main()
