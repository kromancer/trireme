import argparse
import ctypes
from pathlib import Path
from typing import Callable

import jinja2
import numpy as np
import scipy.sparse as sp

from mlir import runtime as rt
from mlir import ir
from mlir.execution_engine import ExecutionEngine

from argument_parsers import get_spmv_arg_parser, add_parser_for_benchmark
from decorators import benchmark, profile, RunFuncType
from common import SparseFormats, make_work_dir_and_cd_to_it
from mlir_exec_engine import create_exec_engine
from generate_kernel import apply_passes, make_spmv_mlir_module
from hwpref_controller import HwprefController
from matrix_storage_manager import create_sparse_mat_and_dense_vec, get_storage_buffers


def get_jinja() -> jinja2.Environment:
    template_dir = Path(__file__).parent.resolve() / "templates"
    return jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))


def render_template_for_main(args: argparse.Namespace) -> str:
    jinja = get_jinja()

    # Function "main" will be injected after the sparse-assembler pass
    main_template = jinja.get_template(f"spmv_{args.matrix_format}.main.mlir.jinja2")
    return main_template.render(rows=args.i, cols=args.j)


def render_template_for_spmv(args: argparse.Namespace) -> str:

    template_names = {"pref-ains": f"spmv_{args.matrix_format}.ains.mlir.jinja2",
                      "pref-spe": f"spmv_{args.matrix_format}.spe.mlir.jinja2"}

    jinja = get_jinja()
    spmv_template = jinja.get_template(template_names[args.optimization])
    spmv_rendered = spmv_template.render(rows=args.i, cols=args.j, pd=args.prefetch_distance,
                                         loc_hint=args.locality_hint)

    return spmv_rendered


def run_spmv(exec_engine: ExecutionEngine, args: argparse.Namespace, mat: sp.sparray, vec: np.ndarray,
             decorator:  Callable[[ExecutionEngine, argparse.Namespace], Callable[[RunFuncType], RunFuncType]]):

    buffers = [ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(b)))
               for b in get_storage_buffers(mat, SparseFormats(args.matrix_format))]

    c_vals = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(vec)))

    a = np.zeros(args.i, np.float64)
    a_vals = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(a)))

    # Allocate a MemRefDescriptor to receive the output tensor.
    ref_out = rt.make_nd_memref_descriptor(1, ctypes.c_double)()
    mem_out = ctypes.pointer(ctypes.pointer(ref_out))

    def run():
        exec_engine.invoke("main", mem_out, *buffers, c_vals, a_vals)
        exec_engine.dump_to_object_file("spmm.o")

        # Sanity check on computed result.
        if args.enable_output_check:
            expected = mat.dot(vec)
            res = rt.ranked_memref_to_numpy(mem_out[0])
            assert np.allclose(res, expected), "Wrong output!"

        # reset output
        a.fill(0)

    decorated_run = decorator(exec_engine, args)(run)
    decorated_run()


def main():
    args = parse_args()
    make_work_dir_and_cd_to_it(__file__)

    with ir.Context(), ir.Location.unknown():
        if not args.optimization or args.optimization == "pref-mlir":
            spmv = str(make_spmv_mlir_module(args.i, args.j, SparseFormats(args.matrix_format)))
            pipeline = "pref" if args.optimization == "pref-mlir" else "no-opt"
        else:
            spmv = render_template_for_spmv(args)
            pipeline = "vect-vl4" if args.optimization == "vect-vl4" else "no-opt"

        with open("spmv.mlir", "w") as f:
            f.write(spmv)

        main_fun = render_template_for_main(args)

        llvm_mlir, _ = apply_passes(spmv, kernel="spmv", pipeline=pipeline, main=main_fun)

        mat, vec = create_sparse_mat_and_dense_vec(args.i, args.j, args.density, form=SparseFormats(args.matrix_format))

        with HwprefController(args):
            if args.command == "benchmark":
                decorator = benchmark
            elif args.command == "profile":
                decorator = profile

            exec_engine = create_exec_engine(llvm_mlir)
            run_spmv(exec_engine, args, mat, vec, decorator=decorator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="(Sparse Matrix)x(Dense Vector) Multiplication (SpMV), "
                                                 "baseline and state-of-the-art sw prefetching, "
                                                 "from manually generated MLIR templates")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # common args to all subcommands
    common_arg_parser = get_spmv_arg_parser()
    common_arg_parser.add_argument("-o", "--optimization",
                                   choices=["vect-vl4", "pref-mlir", "pref-ains", "pref-spe"],
                                   help="Use an optimized version of the kernel")
    HwprefController.add_args(common_arg_parser)

    add_parser_for_benchmark(subparsers, parent_parser=common_arg_parser)

    profile_parser = subparsers.add_parser("profile", parents=[common_arg_parser],
                                           help="Profile the application")
    profile_parser.add_argument("analysis", choices=["toplev", "vtune", "events"],
                                help="Choose an analysis type")

    return parser.parse_args()


if __name__ == "__main__":
    main()
