import argparse
import ctypes
from pathlib import Path

import jinja2
import numpy as np
import scipy.sparse as sp

from mlir import runtime as rt
from mlir.execution_engine import *
from mlir import ir

from argument_parsers import add_parser_for_benchmark, get_spmm_arg_parser
from decorators import benchmark
from common import Encodings, make_work_dir_and_cd_to_it
from matrix_storage_manager import create_sparse_mat
from mlir_exec_engine import create_exec_engine
from generate_kernel import apply_passes, make_spmm_mlir_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="(Sparse Matrix)x(Sparse Matrix) Multiplication (SpMM)")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    common_arg_parser = get_spmm_arg_parser()
    add_parser_for_benchmark(subparsers, parent_parser=common_arg_parser)

    return parser.parse_args()


def run_spmm(exec_engine: ExecutionEngine, args: argparse.Namespace, B: sp.csr_array, C: sp.csr_array):
    B2_pos = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(B.indptr)))
    B2_crd = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(B.indices)))
    B_vals = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(B.data)))

    C2_pos = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(C.indptr)))
    C2_crd = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(C.indices)))
    C_vals = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(C.data)))

    A = np.zeros((args.i, args.k), np.float64)
    A_vals = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(A)))

    # Allocate a MemRefDescriptor to receive the output tensor.
    # The buffer itself is allocated inside the MLIR code generation.
    ref_out = rt.make_nd_memref_descriptor(2, ctypes.c_double)()
    mem_out = ctypes.pointer(ctypes.pointer(ref_out))

    def run():
        exec_engine.invoke("main", mem_out, A_vals, B2_pos, B2_crd, B_vals, C2_pos, C2_crd, C_vals)
        exec_engine.dump_to_object_file("spmm.o")

        # Sanity check on computed result.
        if args.enable_output_check:
            expected = B.dot(C).toarray()
            res = rt.ranked_memref_to_numpy(mem_out[0])
            assert np.allclose(res, expected), "Wrong output!"

        # reset output
        A.fill(0)

    decorated_run = benchmark(exec_engine, args)(run)
    decorated_run()


def render_main_template(rows: int, cols: int) -> str:
    template_dir = Path(__file__).parent.resolve() / "templates"
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
    main_template = env.get_template("spmm_csr_csr.main.mlir.jinja2")
    main_rendered = main_template.render(rows=rows, cols=cols)
    return main_rendered


def main():
    args = parse_args()
    make_work_dir_and_cd_to_it(__file__)

    enc_B = Encodings.CSR
    B = create_sparse_mat(args.i, args.k, args.density, enc_B)

    enc_C = Encodings.CSR
    C = create_sparse_mat(args.k, args.j, args.density, enc_C)

    with ir.Context(), ir.Location.unknown():
        spmm = str(make_spmm_mlir_module(args.i, args.j, args.k, enc_B, enc_C))
        with open(f"spmm.mlir", "w") as f:
            f.write(spmm)
        main = render_main_template(args.i, args.j)
        llvm_mlir, _ = apply_passes(kernel="spmm", src=spmm,
                                    pipeline="pref" if args.enable_prefetches else "no-opt", main=main)

    exec_engine = create_exec_engine(llvm_mlir)
    if args.command == "benchmark":
        run_spmm(exec_engine=exec_engine, args=args, B=B, C=C)


if __name__ == "__main__":
    main()
