from argparse import Namespace
from pathlib import Path
import tarfile
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.io import mmread
import scipy.sparse as sp

from common import change_dir, print_size, read_config, SparseFormats, timeit
from suite_sparse import get_suitesparse_matrix


def get_storage_buffers(mat: sp.sparray, m_form: SparseFormats) -> Tuple[List[np.array], np.dtype, np.dtype]:
    if m_form == SparseFormats.CSR:
        mat: sp.csr_array
        return [mat.indptr, mat.indices, mat.data], mat.data.dtype, mat.indptr.dtype
    elif m_form == SparseFormats.COO:
        mat: sp.coo_array
        pos = np.array([0, mat.nnz], dtype=mat.row.dtype)
        return [pos, mat.row, mat.col, mat.data], mat.data.dtype, mat.row.dtype
    else:
        assert False, "Unknown format"


class InputManager:

    # For type hinting the "args" property
    class ArgsNamespace(Namespace):
        in_source: str
        matrix_format: SparseFormats
        # if input source is synthetic:
        # non-optional fields will be explicitly set by this module if input is SuiteSparse
        i: int
        j: int
        val_type: str
        density: Optional[float]
        # if input source is from SuiteSparse:
        name: Optional[str]

    def __init__(self, args: ArgsNamespace):
        self.args = args

        seed = read_config("input-manager-config.json", "seed")
        if seed is None:
            seed = 5
        self.rng = np.random.default_rng(seed)

        skip_load = read_config("input-manager-config.json", "skip_load")
        if skip_load is None:
            skip_load = True
        self.skip_load = skip_load

        skip_store = read_config("input-manager-config.json", "skip_store")
        if skip_store is None:
            skip_store = True
        self.skip_store = skip_store

        sdir = read_config("input-manager-config.json", "directory")
        if sdir is None:
            sdir = Path.cwd()
        else:
            sdir = Path(sdir)
        self.directory = sdir

    def file_path(self, prefix: str, **kwargs) -> Path:
        params = "_".join(f"{key}-{value}" for key, value in kwargs.items())
        return self.directory / f"{prefix}_{params}.npz"

    @timeit
    def create_sparse_mat(self) -> sp.sparray:
        if self.args.in_source == "synthetic":
            return self.create_synth_sparse_mat()
        else:
            return self.get_ss_mat()

    @timeit
    def create_dense_vec(self) -> np.ndarray:
        size = self.args.j
        vtype = self.args.val_type
        file_path = self.file_path('dense_vector', size=size, vtype=vtype)
        if file_path.exists() and not self.skip_load:
            return np.load(file_path)['arr_0']

        dense_vec: np.ndarray = self.rng.random(size, dtype=np.dtype(vtype))
        if not self.skip_store:
            np.savez_compressed(file_path, dense_vec)

        print(f"vector: size: {print_size(size * np.float64().itemsize)}"
              f"{', saved as' + str(file_path) if not self.skip_store else ''}")

        return dense_vec

    def create_synth_sparse_mat(self) -> Union[sp.coo_array, sp.csr_array]:
        i = self.args.i
        j = self.args.j
        dens = self.args.density
        vtype = self.args.val_type
        m_f = SparseFormats(self.args.matrix_format)
        file_path = self.file_path('sparse_matrix', i=i, j=j, dens=dens, form=m_f, vtype=vtype)
        if not self.skip_load and file_path.exists():
            return sp.load_npz(file_path)
        m: sp.csr_array = sp.random_array((i, j), density=dens, dtype=np.dtype(vtype),
                                          format="csr", random_state=self.rng)
        if m_f == SparseFormats.COO:
            m: sp.coo_array = m.tocoo()
        if not self.skip_store:
            sp.save_npz(file_path, m)
        print(f"sparse matrix: nnz: {m.nnz}, "
              f"nnz size: {print_size(m.nnz * m.dtype.itemsize)}{', saved as' + str(file_path) if not self.skip_store else ''}")
        return m

    def get_ss_mat(self) -> sp.coo_array:
        mtx = self.args.name
        m_f = SparseFormats(self.args.matrix_format)
        file_path = self.directory / f"{mtx}.tar.gz"

        # If skip_store is False, use a temporary dir to download
        download_dir = None if self.skip_store else self.directory
        with change_dir(download_dir):
            if self.skip_load or not file_path.exists():
                get_suitesparse_matrix(self.args.name)
                file_path = Path.cwd() / f"{mtx}.tar.gz"

            # Always switch to a temporary directory for the extraction
            with change_dir():
                with tarfile.open(file_path, "r:gz") as tar:
                    tar.extractall()
                mtx_file = Path(mtx) / (mtx + ".mtx")
                # The COO read will not have sorted indices, so trigger this by coo -> csr conversion
                m = sp.coo_array(mmread(mtx_file)).tocsr()

        self.args.i, self.args.j = m.shape
        self.args.val_type = m.dtype.name
        if m_f == SparseFormats.COO:
            m: sp.coo_array = m.tocoo()
        return m

    def create_sparse_mat_and_dense_vec(self) -> Tuple[Union[sp.coo_array, sp.csr_array], np.ndarray]:
        return self.create_sparse_mat(), self.create_dense_vec()

