import argparse
from subprocess import run, DEVNULL
from tqdm import tqdm

from suite_sparse import get_all_suitesparse_matrix_names

from argument_parsers import add_sparse_format_arg


def main():
    matrix_names = get_all_suitesparse_matrix_names()

    args = parse_args()

    opt = ["-o", args.optimization] if args.optimization else []

    with tqdm(total=len(matrix_names), desc="spmv on SuiteSparse") as pbar:
        for matrix in matrix_names:
            pbar.set_description(f"spmv on {matrix}")
            command = ["python", "spmv.py", *opt, "--check-output", "--matrix-format", f"{args.matrix_format}",
                       "benchmark", "--repetitions", "10", "SuiteSparse", matrix]
            run(command, stdout=DEVNULL, text=True)
            pbar.update(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="(Sparse Matrix)x(Dense Vector) Multiplication (SpMV) with MLIR on "
                                                 "all SuiteSparse matrices.")
    parser.add_argument("-o", "--optimization",
                        choices=["vect-vl4", "pref-mlir", "pref-ains", "pref-spe"],
                                   help="Use an optimized version of the kernel")
    add_sparse_format_arg(parser, "matrix")

    return parser.parse_args()


if __name__ == "__main__":
    main()
