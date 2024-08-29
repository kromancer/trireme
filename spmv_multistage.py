import argparse
from pathlib import Path

from benchmark import add_parser_for_benchmark
from common import (add_parser_for_profile, benchmark_spmv, build_with_cmake, get_spmv_arg_parser,
                    make_work_dir_and_cd_to_it)
from hwpref_controller import HwprefController
from logging_and_graphing import log_execution_times_secs
from input_manager import create_sparse_mat_and_dense_vec
from vtune import profile_spmv_with_vtune


def main():
    src_path = Path(__file__).parent.resolve() / "multistage"
    assert src_path.exists()

    args = parse_args()

    mat, vec = create_sparse_mat_and_dense_vec(rows=args.rows, cols=args.cols, density=args.density, form="csr")

    make_work_dir_and_cd_to_it(__file__)

    cmake_args = [f"-DL2_MSHRS={args.l2_mshrs}"]

    with HwprefController(args):
        if args.command == "benchmark":
            lib = build_with_cmake(cmake_args=cmake_args, target="benchmark-spmv-multistage",
                                   src_path=src_path, is_lib=True)
            exec_times = benchmark_spmv(args, lib, mat, vec)
            log_execution_times_secs(exec_times)
        elif args.command == "profile":
            cmake_args += ["-DPROFILE_WITH_VTUNE=ON"]
            exe = build_with_cmake(cmake_args=cmake_args, target="spmv-multistage", src_path=src_path)
            profile_spmv_with_vtune(exe, mat, vec, args.vtune_cfg)


def parse_args() -> argparse.Namespace:

    # Add common args for all subcommands
    common_arg_parser = get_spmv_arg_parser(with_pd=False, with_loc_hint=False)
    common_arg_parser.add_argument("--l2-mshrs", default=140, type=int,
                                   help="Number of L2 MSHRs")
    HwprefController.add_args(common_arg_parser)

    # Add subcommands
    parser = argparse.ArgumentParser(description='(Sparse Matrix)x(Dense Vector) Multiplication (SpMV) '
                                                 'with multiple prefetching stages.')
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")
    add_parser_for_profile(subparsers, parent_parser=common_arg_parser)
    add_parser_for_benchmark(subparsers, parent_parser=common_arg_parser)

    return parser.parse_args()


if __name__ == "__main__":
    main()
