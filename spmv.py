import argparse
import ctypes
from pathlib import Path
from typing import Callable, List, Union

import jinja2
import numpy as np
from scipy.sparse import coo_array, csr_array

from log_plot import append_placeholder
from mlir import runtime as rt
from mlir import ir
from mlir.execution_engine import ExecutionEngine

from advisor import profile_spmv_with_advisor
from argument_parsers import (add_args_for_benchmark, add_opt_arg, add_synth_tensor_arg, add_output_check_arg,
                              add_sparse_format_arg, add_prefetch_distance_arg, add_locality_hint_arg)
from common import SparseFormats, make_work_dir_and_cd_to_it
from decorators import benchmark, profile, RunFuncType
from generate_kernel import apply_passes, render_template_for_spmv, translate_to_llvm_ir
from hwpref_controller import HwprefController
from input_manager import InputManager, get_storage_buffers
from mlir_exec_engine import create_exec_engine
from np_to_memref import make_nd_memref_descriptor

to_mlir_type = {
    "float64": "f64",
    "float32": "f32",
    "int64": "i64",
    "int32": "i32",
    "bool": "i1"
}


def get_jinja() -> jinja2.Environment:
    template_dir = Path(__file__).parent.resolve() / "templates"
    return jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))


def render_template_for_main(args: argparse.Namespace) -> str:
    jinja = get_jinja()

    # Function "main" will be injected after the sparse-assembler pass
    main_template = jinja.get_template(f"spmv_{args.matrix_format}.main.mlir.jinja2")
    return main_template.render(rows=args.i, cols=args.j, dtype=to_mlir_type[args.dtype])


def run_spmv(exec_engine: ExecutionEngine, args: argparse.Namespace, mat: Union[coo_array, csr_array],
             mat_buffs: List[np.ndarray], vec: np.ndarray, dtype: np.dtype,
             decorator:  Callable[[ExecutionEngine, argparse.Namespace], Callable[[RunFuncType], RunFuncType]],
             itype: np.dtype = np.dtype("int64"),):

    mat_memrefs = [ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(b))) for b in mat_buffs]

    vec_memref = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(vec)))

    # Allocate a buffer to receive the output
    res_buff = (np.ctypeslib.as_ctypes_type(dtype) * args.i)(0)
    res_memref = make_nd_memref_descriptor(1, dtype=dtype, itype=itype)()
    c_dtype = np.ctypeslib.as_ctypes_type(dtype)
    c_itype = np.ctypeslib.as_ctypes_type(itype)
    res_memref.allocated = ctypes.addressof(res_buff)
    res_memref.aligned = ctypes.cast(res_buff, ctypes.POINTER(c_dtype))
    res_memref.offset = c_itype(0)
    res_memref.shape = (c_itype * 1)(args.i)
    res_memref.strides = (c_itype * 1)(1)

    # Allocate a MemRefDescriptor to receive the output
    ref_out = make_nd_memref_descriptor(1, dtype=dtype, itype=itype)()
    mem_out = ctypes.pointer(ctypes.pointer(ref_out))

    def run():
        exec_engine.invoke("main", mem_out, ctypes.pointer(ctypes.pointer(res_memref)), *mat_memrefs, vec_memref)
        exec_engine.dump_to_object_file("spmv.o")

        # Sanity check on computed result.
        if args.check_output:
            expected = mat.dot(vec)
            res = rt.ranked_memref_to_numpy(mem_out[0])
            assert np.allclose(res, expected), "Wrong output!"

        # reset output
        ctypes.memset(ctypes.addressof(res_buff), 0, ctypes.sizeof(res_buff))

    decorated_run = decorator(exec_engine, args)(run)
    decorated_run()


def run_with_jit(args: argparse.Namespace, llvm_mlir: ir.Module, mat: Union[coo_array, csr_array], vec: np.ndarray):
    mat_buffers, dtype, _ = get_storage_buffers(mat, SparseFormats(args.matrix_format))

    if args.action == "benchmark":
        decorator = benchmark
    else:
        decorator = profile

    exec_engine = create_exec_engine(llvm_mlir)

    with HwprefController(args):
        run_spmv(exec_engine, args, mat, mat_buffers, vec, dtype=dtype, decorator=decorator)


def run_with_aot(args: argparse.Namespace, src: Path, mat: Union[coo_array, csr_array], vec: np.ndarray):
    llvm_ir = translate_to_llvm_ir(src, "spmv")
    profile_spmv_with_advisor(args, llvm_ir, mat, vec)


def main():
    append_placeholder()
    args = parse_args()
    make_work_dir_and_cd_to_it(__file__)

    in_man = InputManager(args)
    mat, vec = in_man.create_sparse_mat_and_dense_vec()

    spmv = render_template_for_spmv(args)
    with open("spmv.0. mlir", "w") as f:
        f.write(spmv)

    if args.optimization == "pref-mlir":
        pipeline = "pref"
    elif args.optimization == "pref-mlir-omp":
        pipeline = "pref-omp"
    elif args.optimization == "vect-vl4":
        pipeline = "vect-vl4"
    elif args.optimization == "omp":
        pipeline = "omp"
    else:
        pipeline = "no-opt"

    uses_aot_compiled_spmv = args.action == "profile" and args.analysis == "advisor"
    if uses_aot_compiled_spmv:
        main_fun = None
    else:
        main_fun = render_template_for_main(args)

    with ir.Context(), ir.Location.unknown():
        llvm_mlir, out = apply_passes(args=args, src=spmv, kernel="spmv", pipeline=pipeline, main_fun=main_fun)

    if uses_aot_compiled_spmv:
        run_with_aot(args, out, mat, vec)
    else:
        run_with_jit(args, llvm_mlir, mat, vec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="(Sparse Matrix)x(Dense Vector) Multiplication (SpMV) with MLIR")

    # common arguments
    add_prefetch_distance_arg(parser)
    add_locality_hint_arg(parser)
    parser.add_argument("--sparse-vec", action="store_true", help="Use sparse vector")
    add_opt_arg(parser)
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
    profile_parser.add_argument("analysis", choices=["toplev", "vtune", "advisor", "events"],
                                help="Choose an analysis type")
    profile_parser.add_argument("config", type=str,
                                help="Tool's config, read from the corresponding .json")

    # 2nd level subparsers, matrix type, synthetic or from SuiteSparse
    for p in benchmark_parser, profile_parser:
        matrix_subparser = p.add_subparsers(dest="in_source", help="Choose input matrix source: synthetic or SuiteSparse")

        # Subcommand: synthetic
        synthetic_parser = matrix_subparser.add_parser("synthetic", help="Generate a synthetic matrix")
        add_synth_tensor_arg(synthetic_parser, num_dims=2)

        # Subcommand: SuiteSparse
        ss_parser = matrix_subparser.add_parser("SuiteSparse", help="Use SuiteSparse input matrix")
        ss_parser.add_argument("name", type=str, help="Name of SuiteSparse matrix")

    return parser.parse_args()


if __name__ == "__main__":
    main()
