import argparse
from subprocess import run, DEVNULL

from tqdm import tqdm

from input_manager import InputManager
from suite_sparse import SuiteSparse


def main():
    parser = argparse.ArgumentParser(description="Run kernel on SuiteSparse matrices.",
                                     epilog="Any arguments not listed above will be forwarded as-is "
                                            "to the corresponding <kernel>.py.")
    SuiteSparse.add_args(parser)
    parser.add_argument("--kernel", choices=["spmv", "spmv"], default="spmv")
    args, unknown_args = parser.parse_known_args()

    command = ["python", f"{args.kernel}.py"] + unknown_args

    ss = SuiteSparse(InputManager.get_working_dir("SuiteSparse"), args)
    matrices = ss.get_matrices()

    with tqdm(total=len(matrices), desc=f"{args.kernel} on SuiteSparse") as pbar:
        for m in matrices:
            pbar.set_description(f"{args.kernel} on {m}")
            run(command + ["SuiteSparse", m], stdout=DEVNULL, text=True)
            pbar.update(1)


if __name__ == "__main__":
    main()
