import argparse
import json
from pathlib import Path
from subprocess import run, DEVNULL
from tqdm import tqdm

from common import read_config
from suite_sparse import get_all_suitesparse_matrix_names

from argument_parsers import add_args_for_benchmark, add_output_check_arg, add_sparse_format_arg


def main():
    script_dir = Path(__file__).parent.resolve()
    cfg_file = script_dir / "suite-sparse-config.json"

    args = parse_args(cfg_file)

    opt = ["-o", args.optimization] if args.optimization else []
    check_output = ["--check-output"] if args.check_output else []

    command = ["python", "spmv.py", *check_output, *opt, "--matrix-format", f"{args.matrix_format}"]

    if args.action == "profile":
        command += [args.action, args.analysis]
    else:
        command += ["benchmark", "--repetitions", "10"]

    if args.collection == "all":
        matrix_names = {get_all_suitesparse_matrix_names(is_real=True)}
        matrix_names -= {read_config("suite-sparse-config.json", "exclude-from-all")}
    else:
        matrix_names = read_config("suite-sparse-config.json", args.collection)

    with tqdm(total=len(matrix_names), desc="spmv on SuiteSparse") as pbar:
        for matrix in matrix_names:
            pbar.set_description(f"spmv on {matrix}")
            run(command + ["SuiteSparse", matrix], stdout=DEVNULL, text=True)
            pbar.update(1)


def parse_args(cfg_file: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="(Sparse Matrix)x(Dense Vector) Multiplication (SpMV) with MLIR on "
                                                 "SuiteSparse matrices.")
    cfg = {}
    try:
        with open(cfg_file, "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        print(f"Could not locate {cfg_file}")
    except json.decoder.JSONDecodeError as e:
        print(f"{cfg_file} could not be decoded {e}")

    add_output_check_arg(parser)
    parser.add_argument("-c", "--collection",
                        choices=list(cfg.keys()) + ["all"],
                        help="Specify the collection of SuiteSparse matrices to use for SpMV. "
                             "Choose from predefined collections in "
                             f"{cfg_file}, or use 'all' to run on any matrix that is not in 'exclude-from-all'.")
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
