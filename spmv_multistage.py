import argparse

from pathlib import Path

from create_sparse_mats import create_sparse_mat_and_dense_vec
from logging_and_graphing import log_execution_times_secs
from utils import (benchmark_spmv, build_with_cmake, get_spmv_arg_parser, make_work_dir_and_cd_to_it,
                   profile_spmv_with_vtune)


def main():
    src_path = Path(__file__).parent.resolve() / "multistage"
    assert src_path.exists()

    args = parse_args()

    mat, vec = create_sparse_mat_and_dense_vec(rows=args.rows, cols=args.cols, density=args.density, form="csr")

    make_work_dir_and_cd_to_it(__file__)

    cmake_args = [f"-DL1_MSHRS={args.l1_mshrs}", f"-DL2_MSHRS={args.l2_mshrs}"]
    if args.command == "benchmark":
        lib = build_with_cmake(cmake_args=cmake_args, target="benchmark-spmv-multistage",
                               src_path=src_path, is_lib=True)
        exec_times = benchmark_spmv(args, lib, mat, vec)
        log_execution_times_secs(exec_times)
    elif args.command == "profile":
        exe = build_with_cmake(cmake_args=cmake_args, target="spmv-multistage", src_path=src_path)
        profile_spmv_with_vtune(exe, mat, vec, args.vtune_cfg)


def parse_args() -> argparse.Namespace:
    common_arg_parser = get_spmv_arg_parser(with_pd=False, with_loc_hint=False)
    common_arg_parser.add_argument("--l1-mshrs", default=3, type=int,
                                   help="Number of L1D MSHRs")
    common_arg_parser.add_argument("--l2-mshrs", default=140, type=int,
                                   help="Number of L2 MSHRs")

    parser = argparse.ArgumentParser(description='(Sparse Matrix)x(Dense Vector) Multiplication (SpMV) '
                                                 'with multiple prefetching stages.')
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # TODO: move this to utils.py
    profile_parser = subparsers.add_parser("profile", parents=[common_arg_parser],
                                           help="Profile the application using vtune")
    profile_parser.add_argument("vtune_cfg", choices=["uarch", "threading", "prefetches"],
                                help="Choose an analysis type")

    # TODO: move this to utils.py
    benchmark_parser = subparsers.add_parser("benchmark", parents=[common_arg_parser],
                                             help="Benchmark the application.")
    benchmark_parser.add_argument("--repetitions", type=int, default=5,
                                  help="Repeat the kernel with the same input. Gather execution times stats")

    return parser.parse_args()


if __name__ == "__main__":
    main()