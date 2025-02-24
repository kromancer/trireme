import gc
from pathlib import Path
import tarfile

import scipy.sparse as sp

from common import change_dir, extract_tar, read_config, timeit
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
                print(f"Converting {mtx}")
                out = input_dir / (mtx + ".npz")
                if out.exists():
                    continue

                extract_tar(file)

                rb_file = Path(mtx) / (mtx + ".rb")
                assert rb_file.exists()

                i = ss.get_meta(mtx, "num_of_rows")
                j = ss.get_meta(mtx, "num_of_cols")
                nnz = ss.get_meta(mtx, "num_of_entries")
                if ss.is_binary(mtx):
                    dtype = "bool"
                else:
                    dtype = "float64"

                mat, _ = rbio.read_rb(rb_file, i, j, nnz, dtype)

                @timeit
                def save():
                    sp.save_npz(out, mat.tocsr(copy=False))
                save()

                print(f"Saved {mtx} to {input_dir / mtx}.npz")
                gc.collect()
