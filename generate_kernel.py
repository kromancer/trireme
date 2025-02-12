import argparse
from contextlib import contextmanager
from platform import machine
from os import chdir, environ, getcwd, makedirs
from pathlib import Path
from subprocess import run
from typing import Tuple

import jinja2
import numpy as np

from mlir import ir
from mlir.passmanager import *

from argument_parsers import (add_dimension_args, add_dtype_arg, add_locality_hint_arg, add_opt_arg,
                              add_prefetch_distance_arg, add_sparse_format_arg)
from common import SparseFormats, make_work_dir_and_cd_to_it

pipelines = {
    "no-opt":
    ["sparse-assembler",
     "sparse-reinterpret-map",
     "sparsification{enable-runtime-library=false}",
     "sparse-tensor-codegen",
     "sparse-storage-specifier-to-llvm",
     "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
     "convert-scf-to-cf",
     "expand-strided-metadata",
     "finalize-memref-to-llvm{index-bitwidth=? use-aligned-alloc=false use-generic-functions=false}",
     "convert-func-to-llvm{index-bitwidth=? use-bare-ptr-memref-call-conv=false}",
     "convert-arith-to-llvm{index-bitwidth=?}",
     "convert-index-to-llvm{index-bitwidth=?}",
     "convert-cf-to-llvm{index-bitwidth=?}",
     "reconcile-unrealized-casts"],

    "pref":
    ["sparse-assembler",
     "sparse-reinterpret-map",
     "sparsification{enable-runtime-library=false pd=0}",
     "sparse-tensor-codegen",
     "sparse-storage-specifier-to-llvm",
     "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
     "convert-scf-to-cf",
     "expand-strided-metadata",
     "finalize-memref-to-llvm{index-bitwidth=? use-aligned-alloc=false use-generic-functions=false}",
     "convert-func-to-llvm{index-bitwidth=? use-bare-ptr-memref-call-conv=false}",
     "convert-arith-to-llvm{index-bitwidth=?}",
     "convert-cf-to-llvm{index-bitwidth=?}",
     "reconcile-unrealized-casts"],

    "pref-omp":
    ["sparse-assembler",
     "sparse-reinterpret-map",
     "sparsification{enable-runtime-library=false pd=0 parallelization-strategy=dense-any-loop}",
     "sparse-tensor-codegen",
     "sparse-storage-specifier-to-llvm",
     "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
     "convert-scf-to-openmp",
     "canonicalize",
     "convert-scf-to-cf",
     "expand-strided-metadata",
     "finalize-memref-to-llvm{index-bitwidth=? use-aligned-alloc=false use-generic-functions=false}",
     "convert-func-to-llvm{index-bitwidth=? use-bare-ptr-memref-call-conv=false}",
     "canonicalize",
     "convert-openmp-to-llvm",
     "convert-arith-to-llvm{index-bitwidth=?}",
     "convert-cf-to-llvm{index-bitwidth=?}",
     "reconcile-unrealized-casts"],

    "vect-vl4":
    ["sparse-assembler",
     "sparse-reinterpret-map",
     "sparsification{enable-runtime-library=false}",
     "sparse-vectorization{vl=4}",
     "sparse-tensor-codegen",
     "sparse-storage-specifier-to-llvm",
     "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
     "convert-scf-to-cf",
     "expand-strided-metadata",
     "lower-affine",
     "finalize-memref-to-llvm{index-bitwidth=? use-aligned-alloc=false use-generic-functions=false}",
     f"convert-vector-to-llvm{{{'enable-x86vector' if machine() == 'x86_64' else 'enable-arm-neon'}}}",
     "convert-func-to-llvm{index-bitwidth=? use-bare-ptr-memref-call-conv=false}",
     "convert-arith-to-llvm{index-bitwidth=?}",
     "convert-index-to-llvm{index-bitwidth=?}",
     "convert-cf-to-llvm{index-bitwidth=?}",
     "reconcile-unrealized-casts"],

    "omp":
    ["sparse-assembler",
     "sparse-reinterpret-map",
     "sparsification{enable-runtime-library=false parallelization-strategy=dense-any-loop}",
     "convert-scf-to-openmp",
     "sparse-tensor-codegen",
     "sparse-storage-specifier-to-llvm",
     "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
     "canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}",
     "convert-scf-to-cf",
     "expand-strided-metadata",
     "finalize-memref-to-llvm{index-bitwidth=? use-aligned-alloc=false use-generic-functions=false}",
     "convert-func-to-llvm{index-bitwidth=? use-bare-ptr-memref-call-conv=false}",
     "convert-arith-to-llvm{index-bitwidth=?}",
     "convert-index-to-llvm{index-bitwidth=?}",
     "convert-cf-to-llvm{index-bitwidth=?}",
     "convert-openmp-to-llvm",
     "canonicalize",
     "reconcile-unrealized-casts"]
}

# defer execution by using lambdas, requires an active MLIR "Context"
np_to_mlir_type = {
    np.dtype("float64"): lambda: ir.F64Type.get(),
    np.dtype("float32"): lambda: ir.F32Type.get(),
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


def apply_passes(args: argparse.Namespace, src: str, kernel: str, pipeline: str,
                 index_type: np.dtype = np.dtype("int64")) -> Tuple[ir.Module, Path]:
    out_file_name: str

    def run_pass(mlir_opt_pass: str):
        nonlocal module

        # Adapt the width of the index type
        if "index-bitwidth" in mlir_opt_pass:
            mlir_opt_pass = mlir_opt_pass.replace("index-bitwidth=?",
                                                  f"index-bitwidth={np_to_mlir_type[index_type]().width}")

        # Adapt the prefetch distance
        if "pd" in mlir_opt_pass:
            mlir_opt_pass = mlir_opt_pass.replace("pd=0",
                                                  f"pd={args.prefetch_distance}")

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

    return module, Path(out_file_name)


encodings = {SparseFormats.CSR: "#sparse_tensor.encoding<{ map = (d0, d1) -> (d0: dense, d1: compressed) }>",
             SparseFormats.CSC: "#sparse_tensor.encoding<{ map = (d0, d1) -> (d1: dense, d0: compressed) }>",
             SparseFormats.COO: "#sparse_tensor.encoding<{ map = (d0, d1) -> (d0: compressed(nonunique), d1: singleton(soa)) }>"}


def get_jinja() -> jinja2.Environment:
    template_dir = Path(__file__).parent.resolve() / "templates"
    return jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))


def render_template_for_spmv(args: argparse.Namespace) -> str:
    jinja = get_jinja()

    # Prepare template parameters
    encoding = encodings[SparseFormats(args.matrix_format)]
    dtype = to_mlir_type[args.dtype]
    if args.sparse_vec:
        vtype = f"tensor<{args.j}x{dtype}, #sparse_tensor.encoding<{{ map = (d0) -> (d0 : compressed) }}>>"
    else:
        vtype = f"tensor<{args.j}x{dtype}>"

    mat_type = f"tensor<{args.i}x{args.j}x{dtype}, #sparse>"
    out_type = f"tensor<{args.i}x{dtype}>"

    if dtype == "i1":
        add_op = "arith.ori"
        mul_op = "arith.andi"
    elif dtype.startswith("f"):
        add_op = "arith.addf"
        mul_op = "arith.mulf"
    else:
        add_op = "arith.addi"
        mul_op = "arith.muli"

    template_names = {"no-opt": f"spmv.mlir.jinja2",
                      "vect-vl4": f"spmv.mlir.jinja2",
                      "omp": f"spmv.mlir.jinja2",
                      "pref-mlir": f"spmv.mlir.jinja2",
                      "pref-mlir-omp": f"spmv.mlir.jinja2",
                      "pref-split": f"spmv_{args.matrix_format}.split.mlir.jinja2",
                      "pref-ains": f"spmv_{args.matrix_format}.ains.mlir.jinja2",
                      "pref-spe": f"spmv_{args.matrix_format}.spe.mlir.jinja2"}

    spmv_template = jinja.get_template(template_names[args.optimization])
    if args.optimization in ["no-opt", "pref-mlir"]:
        spmv_rendered = spmv_template.render(encoding=encoding, mat_type=mat_type, vtype=vtype, out_type=out_type,
                                             add_op=add_op, mul_op=mul_op, dtype=dtype)
    else:
        spmv_rendered = spmv_template.render(encoding=encoding, mat_type=mat_type, vtype=vtype, out_type=out_type,
                                             add_op=add_op, mul_op=mul_op, dtype=dtype,
                                             rows=args.i, cols=args.j, pd=args.prefetch_distance,
                                             loc_hint=args.locality_hint)
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


def translate_to_llvm_ir(src: Path, out: str) -> Path:
    mlir_translate = Path(environ['LLVM_PATH']) / "bin/mlir-translate"
    assert mlir_translate.exists()

    out = Path(f"{out}.ll")
    translate_cmd = [str(mlir_translate), "--mlir-to-llvmir", src, "-o", out]
    run(translate_cmd, check=True)
    return out


def generate(args: argparse.Namespace, module: ir.Module, kernel_name: str, translate: bool = False):

    if args.optimization in ["pref-ains", "pref-spe", "pref-split"]:
        pipes = [x for x in pipelines if x not in ["pref", "pref-omp"]]
    else:
        pipes = pipelines

    with make_and_switch_dir(kernel_name):
        with open(f"{kernel_name}.mlir", "w") as f:
            f.write(str(module))

        for p in pipes:
            with make_and_switch_dir(p):
                _, last_output = apply_passes(args, str(module), kernel_name, p)

            if translate:
                _ = translate_to_llvm_ir(Path(p) / last_output, f"{kernel_name}_{p}")


def generate_spmv(args: argparse.Namespace):
    with ir.Context() as ctx, ir.Location.unknown():
        module = ir.Module.parse(render_template_for_spmv(args))
        generate(args, module, f"spmv_{args.matrix_format}" + ("_spvec" if args.sparse_vec else ""), translate=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mlir, from the linalg to the llvm dialect, for given kernel")
    add_sparse_format_arg(parser, "matrix")
    add_opt_arg(parser)
    add_dtype_arg(parser)
    add_locality_hint_arg(parser)
    add_prefetch_distance_arg(parser)

    subparsers = parser.add_subparsers(dest="kernel", help="Which kernel to generate")

    spvv_parser = argparse.ArgumentParser(add_help=False)
    add_dimension_args(spvv_parser, 1)
    subparsers.add_parser("spvv", help="Sparse-Vector X Sparse-Vector (SpVV)",
                          parents=[spvv_parser])

    spmv_parser = argparse.ArgumentParser(add_help=False)
    add_dimension_args(spmv_parser, 2)
    spmv_parser.add_argument("--sparse-vec", action="store_true", help="Use sparse vector")
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
        generate_spmv(args)
    elif args.kernel == "spmm":
        # generate_spmm(args.i, args.j, args.k, SparseFormats.CSR, SparseFormats.CSR)
        # generate_spmm(args.i, args.j, args.k, SparseFormats.COO, SparseFormats.COO)
        pass


if __name__ == "__main__":
    main()
