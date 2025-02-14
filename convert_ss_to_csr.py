from argparse import Namespace
from pathlib import Path
import tarfile

import numpy as np
import scipy.sparse as sp

from common import change_dir, read_config
from rbio import RBio
from suite_sparse import SuiteSparse


if __name__ == "__main__":
    input_dir = Path(read_config("input-manager-config.json", "directory-SuiteSparse"))
    rbio = RBio()
    ss = SuiteSparse(input_dir)

    with change_dir():
        for file in input_dir.glob("*.tar.gz"):
            with tarfile.open(file, "r:gz") as tar:
                mtx = file.name.split('.')[0]
                tar.extractall()

                i = ss.get_meta(mtx, "num_of_rows")
                j = ss.get_meta(mtx, "num_of_cols")
                nnz = ss.get_meta(mtx, "num_of_entries")

                rb_file = Path(mtx) / (mtx + ".rb")
                assert rb_file.exists()

                p_Ap, p_Ai = rbio.read_rb(rb_file, i, j, nnz)

                indptr = np.ctypeslib.as_array(p_Ap, shape=(j + 1,))
                indices = np.ctypeslib.as_array(p_Ai, shape=(nnz,))

                if ss.is_binary(mtx):
                    dtype = "bool"
                else:
                    dtype = "float64"

                data = np.ones(nnz, dtype=dtype)

                try:
                    mat = sp.csc_array((data, indices, indptr), shape=(i, j))
                    sp.save_npz(input_dir / (mtx + ".npz"), mat.tocsr(copy=False))
                    print(f"Saved {mtx} to {input_dir / mtx}.npz")
                except ValueError:
                    print(f"FAILED: {mtx}")
                    rbio.free_pos_buffer()
                    rbio.free_idx_buffer()









