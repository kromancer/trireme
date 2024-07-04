import argparse
import ctypes
from pathlib import Path
from typing import Callable, Optional, Tuple

import jinja2
import numpy as np
import scipy.sparse as sp

from mlir import runtime as rt
from mlir import ir
from mlir.execution_engine import ExecutionEngine

from decorators import add_parser_for_benchmark, benchmark, profile, RunFuncType
from common import Encodings, get_spmv_arg_parser, make_work_dir_and_cd_to_it
from mlir_exec_engine import create_exec_engine
from generate_kernel import apply_passes
from hwpref_controller import HwprefController
from matrix_storage_manager import create_sparse_mat_and_dense_vec


def render_templates(rows: int, cols: int, opt: Optional[str], pd: int, loc_hint: int) -> Tuple[str, str]:
    template_dir = Path(__file__).parent.resolve() / "templates"
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))

    template_names = {"pref-ains": "spmv.ainsworth.mlir.jinja2",
                      "pref-spe": "spmv.spe.mlir.jinja2",
                      "no-opt": "spmv.mlir.jinja2"}

    spmv_template = env.get_template(template_names[opt or "no-opt"])
    with open("spmv.mlir", "w") as f:
        if opt == "pref-ains" or opt == "pref-spe":
            spmv_rendered = spmv_template.render(rows=rows, cols=cols, pd=pd, loc_hint=loc_hint)
        else:
            spmv_rendered = spmv_template.render(rows=rows, cols=cols)
        f.write(spmv_rendered)

    # Function "main" will be injected after the sparse-assembler pass
    main_template = env.get_template("spmv.main.mlir.jinja2")
    main_rendered = main_template.render(rows=rows, cols=cols)

    return spmv_rendered, main_rendered


def run_spmv(exec_engine: ExecutionEngine, args: argparse.Namespace, mat: sp.csr_array, vec: np.ndarray,
             decorator:  Callable[[ExecutionEngine, argparse.Namespace], Callable[[RunFuncType], RunFuncType]]):
    B2_pos = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(mat.indptr)))
    B2_crd = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(mat.indices)))
    B_vals = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(mat.data)))

    c_vals = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(vec)))

    a = np.zeros(args.rows, np.float64)
    a_vals = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(a)))

    # Allocate a MemRefDescriptor to receive the output tensor.
    ref_out = rt.make_nd_memref_descriptor(1, ctypes.c_double)()
    mem_out = ctypes.pointer(ctypes.pointer(ref_out))

    def run():
        nonlocal a, a_vals

        exec_engine.invoke("main", mem_out, B2_pos, B2_crd, B_vals, c_vals, a_vals)
        exec_engine.dump_to_object_file("spmm.o")

        # Sanity check on computed result.
        expected = mat.dot(vec)
        res = rt.ranked_memref_to_numpy(mem_out[0])
        assert np.allclose(res, expected), "Wrong output!"

        # reset output
        a = np.zeros(args.rows, np.float64)
        a_vals = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(a)))

    decorated_run = decorator(exec_engine, args)(run)
    decorated_run()


def main():
    args = parse_args()
    make_work_dir_and_cd_to_it(__file__)
    spmv, main = render_templates(args.rows, args.cols, args.optimization, args.prefetch_distance, args.locality_hint)

    with ir.Context(), ir.Location.unknown():
        llvm_mlir, _ = apply_passes(spmv, kernel="spmv", pipeline="vect-vl4" if args.optimization == "vect-vl4" else "no-opt", main=main)

    mat, vec = create_sparse_mat_and_dense_vec(args.rows, args.cols, args.density, form=Encodings.CSR)

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
                                   choices=["vect-vl4", "pref-ains", "pref-spe"],
                                   help="Use an optimized version of the kernel")
    HwprefController.add_args(common_arg_parser)

    add_parser_for_benchmark(subparsers, parent_parser=common_arg_parser)

    # TODO: re-use fun from common.py
    profile_parser = subparsers.add_parser("profile", parents=[common_arg_parser],
                                           help="Profile the application")
    profile_parser.add_argument("analysis", choices=["toplev", "vtune", "events"],
                                help="Choose an analysis type")

    return parser.parse_args()


if __name__ == "__main__":
    main()
