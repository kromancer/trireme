import argparse

import matplotlib.pyplot as plt
from scipy.sparse import csr_array
from tqdm import tqdm


from input_manager import InputManager
from report_manager import create_report_manager, ReportManager
from suite_sparse import SuiteSparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute matplotlib.pyplot.spy on SuiteSparse matrices",)
    SuiteSparse.add_args(parser)
    ReportManager.add_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    args.in_source = "SuiteSparse"
    args.matrix_format = "csr"
    in_man = InputManager(args)
    rep_man = create_report_manager(args)
    ss = SuiteSparse(InputManager.get_working_dir("SuiteSparse"), args)

    matrix_names = ss.get_matrices()
    with tqdm(total=len(matrix_names), desc="structure score on SuiteSparse") as pbar:
        for mtx in matrix_names:
            rep_man.append_placeholder(mtx)
            args.name = mtx
            m: csr_array = in_man.get_ss_mat()
            plt.spy(m, aspect="auto", aa=True)
            plt.savefig(f"{mtx}.png")
            rep_man.append_result({})
            pbar.update(1)


if __name__ == "__main__":
    main()
