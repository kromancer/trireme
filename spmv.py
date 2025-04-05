import argparse
from pathlib import Path
from platform import system

import jinja2
from mlir import ir
import numpy as np

from argument_parsers import (add_args_for_benchmark, add_opt_arg, add_synth_tensor_arg, add_output_check_arg,
                              add_sparse_format_arg, add_prefetch_distance_arg, add_locality_hint_arg,
                              add_args_for_profile)
from common import build_with_cmake, make_work_dir_and_cd_to_it, np_to_mlir_type, SparseFormats
from generate_kernel import apply_passes, render_template_for_spmv, translate_to_llvm_ir
from hwpref_controller import HwprefController
from input_manager import InputManager, get_storage_buffers
from report_manager import create_report_manager, ReportManager
from run_kernel import run_with_aot

if system() == "Linux":
    from ramdisk_linux import RAMDisk
else:
    assert system() == "Darwin", "Unsupported system!"
    from ramdisk_macos import RAMDisk


def get_jinja() -> jinja2.Environment:
    template_dir = Path(__file__).parent.resolve() / "templates"
    return jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))


def render_template_for_main(args: argparse.Namespace) -> str:
    jinja = get_jinja()

    # Function "main" will be injected after the sparse-assembler pass
    main_template = jinja.get_template(f"spmv_{args.matrix_format}.main.mlir.jinja2")
    return main_template.render(rows=args.i, cols=args.j, dtype=np_to_mlir_type[args.dtype])


def main():
    args = parse_args()

    in_man = InputManager(args)
    rep_man = create_report_manager(args)
    mat, vec = in_man.create_sparse_mat_and_dense_vec()
    mat_buffs, _, itype = get_storage_buffers(mat, SparseFormats(args.matrix_format))
    args.itype = itype

    make_work_dir_and_cd_to_it(__file__)

    spmv = render_template_for_spmv(args)
    with open("spmv.0. mlir", "w") as f:
        f.write(spmv)

    if args.optimization in ["omp", "pref-mlir-omp", "pref-ains-omp"]:
        pipeline = "omp"
    elif args.optimization == "vect-vl4":
        pipeline = "vect-vl4"
    else:
        pipeline = "base"

    with ir.Context(), ir.Location.unknown():
        llvm_mlir, out = apply_passes(args=args, src=spmv, kernel="spmv", pipeline=pipeline, index_type=itype)

    # Translate MLIR's llvm dialect to llvm IR, compile and link
    llvm_ir = translate_to_llvm_ir(out, "spmv").resolve()
    src_path = Path(__file__).parent.resolve() / "templates"
    exe = build_with_cmake(
        [f"-DMAIN_FILE=spmv_{'csx' if args.matrix_format in ['csr', 'csc'] else args.matrix_format}.main.c",
         f"-DINDEX_TYPE={itype}", f"-DKERNEL_LLVM_IR={llvm_ir}"],
        target="main", src_path=src_path)

    expected = None
    if args.check_output:
        expected = mat.dot(vec)
        if args.symmetric:
            # "expected" reflects just L * vec
            # "mat" is symmetric, and is stored as a Lower (L) triangular sparse matrix
            LT_dot_vec = mat.transpose().dot(vec)
            if args.dtype == "bool":
                #  Compute: [ L + L^T ] * vec
                expected = expected + LT_dot_vec
            else:
                #  Compute: [ L + L^T - D(L) ] * vec
                D_dot_vec = mat.diagonal() * vec
                expected = expected + LT_dot_vec - D_dot_vec

    res = np.zeros(args.i, dtype=vec.dtype)
    partial_cmd = [str(exe), str(args.i), str(args.j), str(mat.nnz)]
    run_with_aot(args, partial_cmd, res, mat_buffs, vec, expected, in_man, rep_man)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="(Sparse Matrix)x(Dense Vector) Multiplication (SpMV) with MLIR")

    # common arguments
    add_prefetch_distance_arg(parser)
    add_locality_hint_arg(parser)
    parser.add_argument("--sparse-vec", action="store_true", help="Use sparse vector")
    add_opt_arg(parser)
    add_sparse_format_arg(parser, "matrix")
    add_output_check_arg(parser)
    RAMDisk.add_args(parser)
    HwprefController.add_args(parser)
    ReportManager.add_args(parser)

    # 1st level subparsers, benchmark or profile
    action_subparser = parser.add_subparsers(dest="action", help="Choose action: benchmark or profile")

    # Subcommand: benchmark
    benchmark_parser = action_subparser.add_parser("benchmark")
    add_args_for_benchmark(benchmark_parser)

    # Subcommand: profile
    profile_parser = action_subparser.add_parser("profile")
    add_args_for_profile(profile_parser)

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
