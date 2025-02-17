import gc
from pathlib import Path
import tarfile

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
                out = input_dir / (mtx + ".npz")
                if out.exists():
                    continue

                tar.extractall()

                rb_file = Path(mtx) / (mtx + ".rb")
                assert rb_file.exists()

                i = ss.get_meta(mtx, "num_of_rows")
                j = ss.get_meta(mtx, "num_of_cols")
                nnz = ss.get_meta(mtx, "num_of_entries")
                if ss.is_binary(mtx):
                    dtype = "bool"
                else:
                    dtype = "float64"

                mat = rbio.read_rb(rb_file, i, j, nnz, dtype)
                sp.save_npz(out, mat.tocsr(copy=False))
                print(f"Saved {mtx} to {input_dir / mtx}.npz")
                gc.collect()
