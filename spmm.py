import argparse
from pathlib import Path
from platform import system
import re
from typing import List
from subprocess import run, PIPE

import jinja2
from mlir import ir
import numpy as np
from scipy.sparse import diags_array

from argument_parsers import (add_args_for_benchmark, add_opt_arg, add_synth_tensor_arg, add_output_check_arg,
                              add_sparse_format_arg, add_prefetch_distance_arg, add_locality_hint_arg,
                              add_args_for_profile)
from common import build_with_cmake, flush_cache, make_work_dir_and_cd_to_it, SparseFormats
from generate_kernel import apply_passes, render_template_for_spmm, translate_to_llvm_ir
from hwpref_controller import HwprefController
from input_manager import get_storage_buffers, InputManager
from report_manager import create_report_manager, ReportManager
from spmv import to_mlir_type

if system() == "Linux":
    from ramdisk_linux import RAMDisk
else:
    assert system() == "Darwin", "Unsupported system!"
    from ramdisk_macos import RAMDisk


def get_jinja() -> jinja2.Environment:
    template_dir = Path(__file__).parent.resolve() / "templates"
    return jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))


def render_main_template(args: argparse.Namespace) -> str:
    jinja = get_jinja()

    # Function "main" will be injected after the sparse-assembler pass
    main_template = jinja.get_template("spmm_csr_csr.main.mlir.jinja2")
    main_rendered = main_template.render(rows=args.i, cols=args.j, dtype=to_mlir_type[args.dtype],
                                         dense_output=args.dense_output)
    return main_rendered


def run_with_aot(args: argparse.Namespace, exe: Path, nnz: int, sp_mat_buffs: List[np.array], dense_mat: np.ndarray,
                 exp_out: np.ndarray, in_man: InputManager, rep_man: ReportManager,):
    res = np.zeros((args.i, args.k), dtype=dense_mat.dtype)
    assert res.flags["C_CONTIGUOUS"], "Result matrix must be in row-major order"
    with (RAMDisk(args, in_man, dense_mat, *sp_mat_buffs, res) as ramdisk, HwprefController(args)):
        if args.action == "profile":
            assert False, "TODO: add support for profiling"
        else:
            spmm_cmd = [str(exe), str(args.i), str(args.j), str(args.k), str(nnz)] + ramdisk.buffer_paths

            exec_times = []
            for _ in range(args.repetitions):
                result = run(spmm_cmd, check=True, stdout=PIPE, stderr=PIPE, text=True)

                if args.check_output:
                    assert np.allclose(exp_out, ramdisk.buffers[-1]), "Wrong output!"

                ramdisk.reset_res_buff()
                flush_cache()

                match = re.search(r"Exec time: ([0-9.]+)s", result.stdout)
                assert match is not None, "Execution time not found in the output."
                exec_times.append(float(match.group(1)))

            rep_man.log_execution_times_secs(exec_times)


def main():
    args = parse_args()
    make_work_dir_and_cd_to_it(__file__)

    in_man = InputManager(args)
    rep_man = create_report_manager(args)
    sp_mat, dense_mat = in_man.create_sparse_mat_and_dense_mat()
    sp_mat_buffers, dtype, itype = get_storage_buffers(sp_mat, SparseFormats(args.matrix_format))

    spmm = render_template_for_spmm(args)
    with open("spmm.0. mlir", "w") as f:
        f.write(spmm)

    if args.optimization in ["omp", "pref-mlir-omp", "pref-ains-omp"]:
        pipeline = "omp"
    elif args.optimization == "vect-vl4":
        pipeline = "vect-vl4"
    else:
        pipeline = "base"

    with ir.Context(), ir.Location.unknown():
        llvm_mlir, out = apply_passes(args=args, src=spmm, kernel="spmv", pipeline=pipeline, index_type=itype)

    # Translate MLIR's llvm dialect to llvm IR, compile and link
    llvm_ir = translate_to_llvm_ir(out, "spmm").resolve()
    src_path = Path(__file__).parent.resolve() / "templates"
    exe = build_with_cmake(
        [f"-DMAIN_FILE=spmm_{'csx' if args.matrix_format in ['csr', 'csc'] else args.matrix_format}.main.c",
         f"-DINDEX_TYPE={itype}", f"-DKERNEL_LLVM_IR={llvm_ir}"],
        target="main", src_path=src_path)

    expected = None
    if args.check_output:
        expected = sp_mat.dot(dense_mat)
        if args.symmetric:
            # "expected" reflects just L * dense_mat
            # "sp_mat" is symmetric, and is stored as a Lower (L) triangular sparse matrix
            LT_dot_dense_mat = sp_mat.transpose().dot(dense_mat)
            if args.dtype == "bool":
                #  Compute: [ L + L^T ] * dense_mat
                expected = expected + LT_dot_dense_mat
            else:
                #  Compute: [ L + L^T - D(L) ] * dense_mat
                D_dot_dense_mat = diags_array(sp_mat.diagonal(), offsets=0).dot(dense_mat)
                expected = expected + LT_dot_dense_mat - D_dot_dense_mat

    run_with_aot(args, exe, sp_mat.nnz, sp_mat_buffers, dense_mat, expected, in_man, rep_man)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="(Sparse Matrix)x(Matrix) Multiplication (SpMM)")

    # common arguments
    add_prefetch_distance_arg(parser)
    add_locality_hint_arg(parser)
    add_opt_arg(parser)
    add_sparse_format_arg(parser, "matrix")
    add_output_check_arg(parser)
    RAMDisk.add_args(parser)
    HwprefController.add_args(parser)
    ReportManager.add_args(parser)
    parser.add_argument("-k", type=int, default=2, help=f"Width of dense mat")

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
