import argparse
from pathlib import Path
from platform import system
import re
from typing import List
from subprocess import run, PIPE

import jinja2
import numpy as np

from log_plot import append_placeholder
from mlir import ir

from argument_parsers import (add_args_for_benchmark, add_opt_arg, add_synth_tensor_arg, add_output_check_arg,
                              add_sparse_format_arg, add_prefetch_distance_arg, add_locality_hint_arg,
                              add_args_for_profile)
from common import build_with_cmake, make_work_dir_and_cd_to_it, SparseFormats
from generate_kernel import apply_passes, render_template_for_spmv, translate_to_llvm_ir
from hwpref_controller import HwprefController
from input_manager import InputManager, get_storage_buffers
from log_plot import log_execution_times_secs
from prof import profile_spmv

if system() == "Linux":
    from ramdisk_linux import RAMDisk
else:
    assert system() == "Darwin", "Unsupported system!"
    from ramdisk_macos import RAMDisk

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


def run_with_aot(args: argparse.Namespace, exe: Path, nnz: int, mat_buffs: List[np.array], vec: np.ndarray,
                 exp_out: np.ndarray):
    res = np.zeros(args.i, dtype=vec.dtype)
    with (RAMDisk(args, vec, *mat_buffs, res) as ramdisk, HwprefController(args)):
        if args.action == "profile":
            profile_spmv(args, exe, nnz, ramdisk.buffer_paths)
            if args.check_output:
                assert np.allclose(exp_out, ramdisk.buffers[-1]), "Wrong output!"
        else:
            spmv_cmd = [str(exe), str(args.i), str(args.j), str(nnz)] + ramdisk.buffer_paths

            exec_times = []
            for _ in range(args.repetitions):
                result = run(spmv_cmd, check=True, stdout=PIPE, stderr=PIPE, text=True)

                if args.check_output:
                    assert np.allclose(exp_out, ramdisk.buffers[-1]), "Wrong output!"

                ramdisk.reset_res_buff()

                match = re.search(r"Exec time: ([0-9.]+)s", result.stdout)
                assert match is not None, "Execution time not found in the output."
                exec_times.append(float(match.group(1)))

            log_execution_times_secs(exec_times)


def main():
    append_placeholder()
    args = parse_args()
    make_work_dir_and_cd_to_it(__file__)

    in_man = InputManager(args)
    mat, vec = in_man.create_sparse_mat_and_dense_vec()
    mat_buffs, _, itype = get_storage_buffers(mat, SparseFormats(args.matrix_format))

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

    with ir.Context(), ir.Location.unknown():
        llvm_mlir, out = apply_passes(args=args, src=spmv, kernel="spmv", pipeline=pipeline, index_type=itype)

    # Translate MLIR's llvm dialect to llvm IR, compile and link
    llvm_ir = translate_to_llvm_ir(out, "spmv").resolve()
    src_path = Path(__file__).parent.resolve() / "templates"
    exe = build_with_cmake([f"-DMAIN_FILE=spmv_{args.matrix_format}.main.c",
                            f"-DINDEX_TYPE={itype}",
                            f"-DKERNEL_LLVM_IR={llvm_ir}"],
                           target="main", src_path=src_path)

    expected = None
    if args.check_output:
        expected = mat.dot(vec)
    run_with_aot(args, exe, mat.nnz, mat_buffs, vec, expected)


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
