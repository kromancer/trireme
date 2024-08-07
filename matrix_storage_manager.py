from pathlib import Path
import tempfile
from typing import List, Tuple, Union

import numpy as np
import scipy.sparse as sp

from common import SparseFormats, timeit, print_size, read_config
from singleton_metaclass import SingletonMeta


def create_sparse_mat(rows: int, cols: int, density: float, form: SparseFormats = SparseFormats.COO) -> sp.sparray:
    return MatrixStorageManager().create_sparse_mat(rows, cols, density, form)


def get_storage_buffers(mat: sp.sparray, format: SparseFormats) -> List[np.array]:
    if format == SparseFormats.CSR:
        mat: sp.csr_array
        return [mat.indptr, mat.indices, mat.data]
    elif format == SparseFormats.COO:
        mat: sp.coo_array
        pos = np.array([0, mat.nnz])
        return [pos, mat.row, mat.col, mat.data]
    else:
        assert False, "Unknown format"


def create_sparse_mat_and_dense_vec(rows: int, cols: int, density: float,
                                    form: SparseFormats = SparseFormats.COO) -> Tuple[sp.sparray, np.ndarray]:
    m = MatrixStorageManager()
    return m.create_sparse_mat(rows, cols, density, form), m.create_dense_vec(cols)


class MatrixStorageManager(metaclass=SingletonMeta):

    def __init__(self, sdir: Path = None, seed: int = None, skip_load: bool = None):
        if seed is None:
            seed = read_config("matrix-storage-manager-config.json", "seed")
            if seed is None:
                seed = 5
        print(f"Using seed: {seed}")
        self.rng = np.random.default_rng(seed)

        if sdir is None:
            sdir = read_config("matrix-storage-manager-config.json", "directory")
            if sdir is not None and Path(sdir).exists():
                sdir = Path(sdir)
            else:
                self.directory = Path(tempfile.mkdtemp())
        print(f"Using storage dir: {sdir}")
        self.directory = sdir

        if skip_load is None:
            skip_load = read_config("matrix-storage-manager-config.json", "skip_load")
            if skip_load is None:
                skip_load = False
        print(f"Skip loading input from storage? {skip_load}")
        self.skip_load = skip_load

    def _file_path(self, prefix: str, **kwargs) -> Path:
        params = "_".join(f"{key}-{value}" for key, value in kwargs.items())
        return self.directory / f"{prefix}_{params}.npz"

    @timeit
    def create_sparse_mat(self, rows: int, cols: int, density: float, form: SparseFormats = SparseFormats.COO) -> Union[sp.coo_array, sp.csr_array]:

        file_path = self._file_path('sparse_matrix', rows=rows, cols=cols, density=density, format=form)
        if file_path.exists() and not self.skip_load:
            return sp.load_npz(file_path)

        m: sp.csr_array = sp.random_array((rows, cols), density=density, dtype=np.float64, format="csr", random_state=self.rng)

        if form == SparseFormats.COO:
            m = m.tocoo()

        if not self.skip_load:
            sp.save_npz(file_path, m)

        print(f"sparse matrix: non-zero values size: {print_size(rows * cols * density * np.float64().itemsize)}, "
              f"density: {density * 100}%{', saved as' + str(file_path) if not self.skip_load else ''}")

        return m

    @timeit
    def create_dense_vec(self, size: int) -> np.ndarray:
        file_path = self._file_path('dense_vector', length=size)
        if file_path.exists():
            return np.load(file_path)['arr_0']
        dense_vec = self.rng.random(size)
        np.savez_compressed(file_path, dense_vec)
        print(f"vector: size: {print_size(size * np.float64().itemsize)}{', saved as' + str(file_path) if not self.skip_load else ''}")

        return dense_vec
