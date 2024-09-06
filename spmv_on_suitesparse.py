import argparse
from subprocess import run, DEVNULL
from tqdm import tqdm

from suite_sparse import get_all_suitesparse_matrix_names

from argument_parsers import add_args_for_benchmark, add_sparse_format_arg


def main():
    matrix_names = get_all_suitesparse_matrix_names(is_real=True)

    args = parse_args()

    opt = ["-o", args.optimization] if args.optimization else []

    command = ["python", "spmv.py", *opt, "--check-output", "--matrix-format", f"{args.matrix_format}"]

    if args.action == "profile":
        command += [args.action, args.analysis]
    else:
        command += ["benchmark", "--repetitions", "10"]

    with tqdm(total=len(matrix_names), desc="spmv on SuiteSparse") as pbar:
        for matrix in matrix_names:
            pbar.set_description(f"spmv on {matrix}")
            run(command + ["SuiteSparse", matrix], stdout=DEVNULL, text=True)
            pbar.update(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="(Sparse Matrix)x(Dense Vector) Multiplication (SpMV) with MLIR on "
                                                 "all SuiteSparse matrices.")
    parser.add_argument("-o", "--optimization",
                        choices=["vect-vl4", "pref-mlir", "pref-ains", "pref-spe"],
                                   help="Use an optimized version of the kernel")
    add_sparse_format_arg(parser, "matrix")

    # 1st level subparsers, benchmark or profile
    action_subparser = parser.add_subparsers(dest="action", help="Choose action: benchmark or profile")

    # Subcommand: benchmark
    benchmark_parser = action_subparser.add_parser("benchmark", help="Benchmark the application")
    add_args_for_benchmark(benchmark_parser)

    # Subcommand: profile
    profile_parser = action_subparser.add_parser("profile", help="Profile the application")
    profile_parser.add_argument("analysis", choices=["toplev", "vtune", "events"],
                                help="Choose an analysis type")

    return parser.parse_args()


if __name__ == "__main__":
    main()
