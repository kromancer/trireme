from argparse import Namespace

from tqdm import tqdm

from input_manager import InputManager
from suite_sparse import SuiteSparse


if __name__ == "__main__":
    args = Namespace(matrix_format="csr")
    in_man = InputManager(args)
    mtx_names = SuiteSparse(InputManager.get_working_dir("SuiteSparse")).get_all_matrix_names()

    with tqdm(total=len(mtx_names), desc="spmv on SuiteSparse") as pbar:
        for m in mtx_names:
            pbar.set_description(f"{m}")
            args.name = m
            in_man.get_ss_mat()
            pbar.update(1)

