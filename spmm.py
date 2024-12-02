import argparse
import ctypes
from pathlib import Path
from typing import Callable, List, Union

import jinja2
import numpy as np
from scipy.sparse import coo_array, csr_array

from mlir import runtime as rt
from mlir.execution_engine import *
from mlir import ir

from argument_parsers import add_args_for_benchmark, add_output_check_arg, add_sparse_format_arg, add_synth_tensor_arg
from common import SparseFormats, make_work_dir_and_cd_to_it
from decorators import benchmark, profile, RunFuncType
from generate_kernel import apply_passes, make_spmm_mlir_module
from hwpref_controller import HwprefController
from input_manager import get_storage_buffers, InputManager
from log_plot import append_placeholder
from mlir_exec_engine import create_exec_engine
from np_to_memref import make_nd_memref_descriptor
from spmv import to_mlir_type


def run_spmm(exec_engine: ExecutionEngine, args: argparse.Namespace,
             mat: Union[coo_array, csr_array],
             mat_buffs: List[np.ndarray], dtype: np.dtype, itype: np.dtype,
             decorator:  Callable[[ExecutionEngine, argparse.Namespace], Callable[[RunFuncType], RunFuncType]]):

    mat_memrefs = [ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(b))) for b in mat_buffs]

    # Allocate a buffer to receive the output
    res_buff = (np.ctypeslib.as_ctypes_type(dtype) * (args.i * args.j))(0)
    res_memref = make_nd_memref_descriptor(2, dtype=dtype, itype=itype)()
    c_dtype = np.ctypeslib.as_ctypes_type(dtype)
    c_itype = np.ctypeslib.as_ctypes_type(itype)
    res_memref.allocated = ctypes.addressof(res_buff)
    res_memref.aligned = ctypes.cast(res_buff, ctypes.POINTER(c_dtype))
    res_memref.offset = c_itype(0)
    res_memref.shape = (c_itype * 2)(args.i, args.j)
    res_memref.strides = (c_itype * 2)(1, 1)

    # Allocate a MemRefDescriptor to receive the output
    ref_out = make_nd_memref_descriptor(2, dtype=dtype, itype=itype)()
    mem_out = ctypes.pointer(ctypes.pointer(ref_out))

    def run():
        exec_engine.invoke("main", mem_out, ctypes.pointer(ctypes.pointer(res_memref)),
                           *mat_memrefs, *mat_memrefs)
        exec_engine.dump_to_object_file("spmm.o")

        # Sanity check on computed result.
        if args.check_output:
            expected = mat * mat
            res = rt.ranked_memref_to_numpy(mem_out[0])
            assert np.allclose(res, expected.toarray()), "Wrong output!"

        # reset output
        ctypes.memset(ctypes.addressof(res_buff), 0, ctypes.sizeof(res_buff))

    decorated_run = decorator(exec_engine, args)(run)
    decorated_run()


def get_jinja() -> jinja2.Environment:
    template_dir = Path(__file__).parent.resolve() / "templates"
    return jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))


def render_main_template(args: argparse.Namespace) -> str:
    jinja = get_jinja()

    # Function "main" will be injected after the sparse-assembler pass
    main_template = jinja.get_template("spmm_csr_csr.main.mlir.jinja2")
    main_rendered = main_template.render(rows=args.i, cols=args.j, dtype=to_mlir_type[args.dtype])
    return main_rendered


def main():
    append_placeholder()
    args = parse_args()
    make_work_dir_and_cd_to_it(__file__)

    in_man = InputManager(args)
    mat = in_man.create_sparse_mat()
    mat_format = SparseFormats(args.matrix_format)
    mat_buffers, dtype, itype = get_storage_buffers(mat, mat_format)

    with ir.Context(), ir.Location.unknown():
        if not args.optimization or args.optimization == "pref-mlir":
            spmm = str(make_spmm_mlir_module(args.i, args.j, args.j, mat_format, mat_format, dtype))
            pipeline = "pref" if args.optimization == "pref-mlir" else "no-opt"
        else:
            assert False, "Unsupported optimization"

        with open(f"spmm.no_main.mlir", "w") as f:
            f.write(spmm)

        main_fun = render_main_template(args)
        with open(f"main.mlir", "w") as f:
            f.write(main_fun)

        llvm_mlir, _ = apply_passes(kernel="spmm", src=spmm, pipeline=pipeline, main_fun=main_fun, index_type=itype)

        with HwprefController(args):
            if args.action == "benchmark":
                decorator = benchmark
            elif args.action == "profile":
                decorator = profile

            exec_engine = create_exec_engine(llvm_mlir)
            run_spmm(exec_engine, args, mat, mat_buffers, dtype=dtype, itype=itype, decorator=decorator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="(Sparse Matrix)x(Sparse Matrix) Multiplication (SpMM)")

    # common arguments
    parser.add_argument("-o", "--optimization",
                        choices=["vect-vl4", "pref-mlir", "pref-ains", "pref-spe"],
                        help="Use an optimized version of the kernel")
    add_sparse_format_arg(parser, "matrix")
    add_output_check_arg(parser)
    HwprefController.add_args(parser)

    # 1st level subparsers, benchmark or profile
    action_subparser = parser.add_subparsers(dest="action", help="Choose action: benchmark or profile")

    # Subcommand: benchmark
    benchmark_parser = action_subparser.add_parser("benchmark")
    add_args_for_benchmark(benchmark_parser)

    # Subcommand: profile
    profile_parser = action_subparser.add_parser("profile")
    profile_parser.add_argument("analysis", choices=["toplev", "vtune", "events"],
                                help="Choose an analysis type")

    # 2nd level subparsers, matrix type, synthetic or from SuiteSparse
    for p in benchmark_parser, profile_parser:
        matrix_subparser = p.add_subparsers(dest="in_source",
                                            help="Choose input matrix source: synthetic or SuiteSparse")

        # Subcommand: synthetic
        synthetic_parser = matrix_subparser.add_parser("synthetic", help="Generate a synthetic matrix")
        add_synth_tensor_arg(synthetic_parser, num_dims=2)

        # Subcommand: SuiteSparse
        ss_parser = matrix_subparser.add_parser("SuiteSparse", help="Use SuiteSparse input matrix")
        ss_parser.add_argument("name", type=str, help="Name of SuiteSparse matrix")

    return parser.parse_args()


if __name__ == "__main__":
    main()
