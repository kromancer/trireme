import argparse
import json
from pathlib import Path
from subprocess import run, DEVNULL
from typing import List, Tuple

from tqdm import tqdm

from common import read_config
from suite_sparse import get_all_suitesparse_matrix_names


def main():
    script_dir = Path(__file__).parent.resolve()
    cfg_file = script_dir / "suite-sparse-config.json"

    args, unknown_args = parse_args(cfg_file)

    command = ["python", "spmv.py"] + unknown_args

    if args.collection == "all":
        matrix_names = set(get_all_suitesparse_matrix_names(is_real=True))
        matrix_names -= set(read_config("suite-sparse-config.json", "exclude-from-all"))
    else:
        matrix_names = read_config("suite-sparse-config.json", args.collection)

    with tqdm(total=len(matrix_names), desc="spmv on SuiteSparse") as pbar:
        for matrix in matrix_names:
            pbar.set_description(f"spmv on {matrix}")
            run(command + ["SuiteSparse", matrix], stdout=DEVNULL, text=True)
            pbar.update(1)


def parse_args(cfg_file: Path) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="(Sparse Matrix)x(Dense Vector) Multiplication (SpMV) with MLIR on "
                                                 "SuiteSparse matrices.",
                                     epilog="Any arguments not listed above will be forwarded as-is to spmv.py.")
    cfg = {}
    try:
        with open(cfg_file, "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        print(f"Could not locate {cfg_file}")
    except json.decoder.JSONDecodeError as e:
        print(f"{cfg_file} could not be decoded {e}")

    parser.add_argument("-c", "--collection",
                        choices=list(cfg.keys()) + ["all"],
                        help="Specify the collection of SuiteSparse matrices to use for SpMV. "
                             "Choose from predefined collections in "
                             f"{cfg_file}, or use 'all' to run on any matrix that is not in 'exclude-from-all'.")

    return parser.parse_known_args()


if __name__ == "__main__":
    main()
