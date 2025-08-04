import argparse
from contextlib import contextmanager
from platform import machine
from os import chdir, environ, getcwd, makedirs
from pathlib import Path
from subprocess import run
from typing import Tuple, Union

import jinja2
import numpy as np

from mlir import ir
from mlir.passmanager import *

from argument_parsers import (add_dimension_args, add_dtype_arg, add_itype_arg, add_locality_hint_arg, add_opt_arg,
                              add_prefetch_distance_arg, add_sparse_format_arg)
from common import SparseFormats, make_work_dir_and_cd_to_it

pipelines = {
    "base":
    ["sparse-assembler",
     "sparse-reinterpret-map",
     "sparsifier{enable-runtime-library=false enable-index-optimizations=true pd=?}"],

    "omp":
    ["sparse-assembler",
     "sparse-reinterpret-map",
     "sparsification{enable-runtime-library=false pd=? parallelization-strategy=dense-outer-loop}",
     "loop-invariant-code-motion",
     "sparse-tensor-codegen",
     "sparse-storage-specifier-to-llvm",
     "cse",
     "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
     "convert-scf-to-openmp",
     "canonicalize",
     "convert-scf-to-cf",
     "expand-strided-metadata",
     "convert-openmp-to-llvm{index-bitwidth=?}",
     "canonicalize",
     "reconcile-unrealized-casts"],

    "vect-vl4":
    ["sparse-assembler",
     "sparse-reinterpret-map",
     f"sparsifier{{{'enable-x86vector' if machine() == 'x86_64' else 'enable-arm-neon'} vl=4 enable-runtime-library=false enable-index-optimizations=true pd=?}}"]

    # "vect-vl4":
    # ["sparse-assembler",
    #  "sparse-reinterpret-map",
    #  "sparsification{enable-runtime-library=false}",
    #  "loop-invariant-code-motion",
    #  "sparse-vectorization{vl=4}",
    #  "sparse-tensor-codegen",
    #  "sparse-storage-specifier-to-llvm",
    #  "cse",
    #  "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
    #  "convert-scf-to-cf",
    #  "expand-strided-metadata",
    #  "lower-affine",
    #  f"convert-vector-to-llvm{{{'enable-x86vector' if machine() == 'x86_64' else 'enable-arm-neon'} index-bitwidth=?}}",
    #  "convert-func-to-llvm{index-bitwidth=? use-bare-ptr-memref-call-conv=false}",
    #  "finalize-memref-to-llvm{index-bitwidth=? use-aligned-alloc=false use-generic-functions=false}",
    #  "convert-arith-to-llvm{index-bitwidth=?}",
    #  "convert-index-to-llvm{index-bitwidth=?}",
    #  "convert-cf-to-llvm{index-bitwidth=?}",
    #  "reconcile-unrealized-casts"]
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
                 index_type: np.dtype = np.dtype("int32")) -> Tuple[ir.Module, Path]:
    out_file_name: str

    def run_pass(mlir_opt_pass: str):
        nonlocal module

        # Adapt the width of the index type
        if "index-bitwidth" in mlir_opt_pass:
            mlir_opt_pass = mlir_opt_pass.replace("index-bitwidth=?",
                                                  f"index-bitwidth={np_to_mlir_type[index_type]().width}")

        # Adapt the prefetch distance
        if "pd" in mlir_opt_pass:
            mlir_opt_pass = mlir_opt_pass.replace("pd=?",
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


def get_encoding(form: SparseFormats, index_type: Union[np.dtype, str]) -> str:
    if isinstance(index_type, str):
        index_type = np.dtype(index_type)

    with ir.Context() as ctx, ir.Location.unknown():
        bitwidth = np_to_mlir_type[index_type]().width
    encodings = {SparseFormats.CSR: f"#sparse_tensor.encoding<{{ map = (d0, d1) -> (d0: dense, d1: compressed), "
                                    f"posWidth={bitwidth}, crdWidth={bitwidth} }}>",
                 SparseFormats.CSC: f"#sparse_tensor.encoding<{{ map = (d0, d1) -> (d1: dense, d0: compressed), "
                                    f"posWidth={bitwidth}, crdWidth={bitwidth} }}>",
                 SparseFormats.COO: f"#sparse_tensor.encoding<{{ map = (d0, d1) -> (d0: compressed(nonunique), d1: singleton(soa)), "
                                    f"posWidth={bitwidth}, crdWidth={bitwidth} }}>"}
    return encodings[form]


def get_jinja() -> jinja2.Environment:
    template_dir = Path(__file__).parent.resolve() / "templates"
    return jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))


def get_semiring(dtype: str) -> Tuple[str, str]:
    if dtype == "i1":
        add_op = "arith.ori"
        mul_op = "arith.andi"
    elif dtype.startswith("f"):
        add_op = "arith.addf"
        mul_op = "arith.mulf"
    else:
        add_op = "arith.addi"
        mul_op = "arith.muli"

    return add_op, mul_op


def render_template_for_spmv(args: argparse.Namespace) -> str:
    jinja = get_jinja()

    # Prepare template parameters
    encoding = get_encoding(SparseFormats(args.matrix_format), args.itype)
    dtype = to_mlir_type[args.dtype]
    itype = to_mlir_type[args.itype.name if isinstance(args.itype, np.dtype) else args.itype]

    if args.sparse_vec:
        vtype = f"tensor<{args.j}x{dtype}, #sparse_tensor.encoding<{{ map = (d0) -> (d0 : compressed) }}>>"
    else:
        vtype = f"tensor<{args.j}x{dtype}>"

    mat_type = f"tensor<{args.i}x{args.j}x{dtype}, #sparse>"
    out_type = f"tensor<{args.i}x{dtype}>"

    add_op, mul_op = get_semiring(dtype)

    template_names = {"no-opt": f"spmv.mlir.jinja2",
                      "vect-vl4": f"spmv.mlir.jinja2",
                      "omp": f"spmv.mlir.jinja2",
                      "pref-mlir": f"spmv.mlir.jinja2",
                      "pref-mlir-omp": f"spmv.mlir.jinja2",
                      "pref-ains": f"spmv_{args.matrix_format}.ains.mlir.jinja2",
                      "pref-ains-omp": f"spmv_{args.matrix_format}.ains.mlir.jinja2",
                      "pref-spe": f"spmv_{args.matrix_format}.spe.mlir.jinja2"}

    spmv_template = jinja.get_template(template_names[args.optimization])
    spmv_rendered = spmv_template.render(encoding=encoding, mat_type=mat_type, vtype=vtype, out_type=out_type,
                                         add_op=add_op, mul_op=mul_op, dtype=dtype, itype=itype, rows=args.i,
                                         cols=args.j, pd=args.prefetch_distance, loc_hint=args.locality_hint,
                                         is_symmetric=args.symmetric)
    return spmv_rendered


def render_template_for_spmm(args: argparse.Namespace) -> str:
    jinja = get_jinja()

    # Prepare template parameters
    encoding = get_encoding(SparseFormats(args.matrix_format), np.dtype(args.itype))
    dtype = to_mlir_type[args.dtype]

    dense_mat_type = f"tensor<{args.j}x{args.k}x{dtype}>"
    sp_mat_type = f"tensor<{args.i}x{args.j}x{dtype}, #sparse>"
    out_type = f"tensor<{args.i}x{args.k}x{dtype}>"

    add_op, mul_op = get_semiring(dtype)

    template_names = {"no-opt": f"spmm.mlir.jinja2",
                      "vect-vl4": f"spmm.mlir.jinja2",
                      "pref-mlir-vect-vl4": f"spmm.mlir.jinja2",
                      "omp": f"spmm.mlir.jinja2",
                      "pref-mlir": f"spmm.mlir.jinja2",
                      "pref-mlir-omp": f"spmm.mlir.jinja2"}

    spmm_template = jinja.get_template(template_names[args.optimization])
    spmv_rendered = spmm_template.render(encoding=encoding, sp_mat_type=sp_mat_type, dense_mat_type=dense_mat_type,
                                         out_type=out_type, add_op=add_op, mul_op=mul_op, dtype=dtype,
                                         rows=args.i, cols=args.j,
                                         is_symmetric=args.symmetric)
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

    with make_and_switch_dir(kernel_name):
        with open(f"{kernel_name}.mlir", "w") as f:
            f.write(str(module))

        for p in pipelines:
            with make_and_switch_dir(p):
                _, last_output = apply_passes(args, str(module), kernel_name, p)

            if translate:
                _ = translate_to_llvm_ir(Path(p) / last_output, f"{kernel_name}_{p}")


def generate_spmv(args: argparse.Namespace):
    with ir.Context() as ctx, ir.Location.unknown():
        module = ir.Module.parse(render_template_for_spmv(args))
        generate(args, module, f"spmv_{args.matrix_format}" + ("_spvec" if args.sparse_vec else ""), translate=True)


def generate_spmm(args: argparse.Namespace):
    with ir.Context() as ctx, ir.Location.unknown():
        module = ir.Module.parse(render_template_for_spmm(args))
        generate(args, module, f"spmm_{args.matrix_format}", translate=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mlir, from the linalg to the llvm dialect, for given kernel")
    add_sparse_format_arg(parser, "matrix")
    add_opt_arg(parser)
    add_dtype_arg(parser, "--dtype", "Data type")
    add_itype_arg(parser, "--itype", "Index type")
    add_locality_hint_arg(parser)
    add_prefetch_distance_arg(parser)
    parser.add_argument("--symmetric", action="store_true",
                        help="Assume that the sparse matrix is symmetric")

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

    if args.kernel == "spmv":
        generate_spmv(args)
    elif args.kernel == "spmm":
        generate_spmm(args)
    else:
        assert False, "Kernel not supported"


if __name__ == "__main__":
    main()
