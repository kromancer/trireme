from argparse import Namespace
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import scipy.sparse as sp
sp._matrix_io.PICKLE_KWARGS["allow_pickle"] = True

from common import change_dir, extract_tar, print_size, read_config, SparseFormats, timeit
from rbio import RBio
from suite_sparse import SuiteSparse


def get_storage_buffers(mat: sp.sparray, m_form: SparseFormats) -> Tuple[List[np.array], np.dtype, np.dtype]:
    if m_form == SparseFormats.CSR:
        mat: sp.csr_array
        return [mat.indptr, mat.indices, mat.data], mat.data.dtype, mat.indptr.dtype
    elif m_form == SparseFormats.COO:
        mat: sp.coo_array
        pos = np.array([0, mat.nnz], dtype=mat.row.dtype)
        return [pos, mat.row, mat.col, mat.data], mat.data.dtype, mat.row.dtype
    elif m_form == SparseFormats.CSC:
        mat: sp.csc_matrix
        return [mat.indptr, mat.indices, mat.data], mat.data.dtype, mat.indptr.dtype
    else:
        assert False, "Unknown format"


class InputManager:

    def __init__(self, args: Namespace):
        self.args = args
        self.rbio = None

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

        self.directory = InputManager.get_working_dir(self.args.in_source)

    @staticmethod
    def get_working_dir(in_source: str):
        sdir = read_config("input-manager-config.json", f"directory-{in_source}")
        if sdir is None:
            sdir = Path.cwd()
        else:
            sdir = Path(sdir)
        return sdir

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
        dense_vec = np.ones(self.args.j, dtype=self.args.dtype)
        print(f"vector size: {print_size(self.args.j * np.dtype(self.args.dtype).itemsize)}")
        return dense_vec

    def create_synth_sparse_mat(self) -> Union[sp.coo_array, sp.csr_array]:
        def print_sparse_mat_info(m):
            print(f"sparse matrix nnz: {m.nnz}, "
                  f"nnz size: {print_size(m.nnz * m.dtype.itemsize)}"
                  f"{', saved as' + str(file_path) if not self.skip_store else ''}")

        i = self.args.i
        j = self.args.j
        dens = self.args.density
        vtype = self.args.dtype
        m_f = SparseFormats(self.args.matrix_format)
        file_path = self.file_path('sparse_matrix', i=i, j=j, dens=dens, form=m_f, vtype=vtype)
        if not self.skip_load and file_path.exists():
            m = sp.load_npz(file_path)
            print_sparse_mat_info(m)
            return m
        m: sp.csr_array = sp.random_array((i, j), density=dens, dtype=np.dtype(vtype),
                                          format="csr", random_state=self.rng)
        print_sparse_mat_info(m)
        if m_f == SparseFormats.COO:
            m: sp.coo_array = m.tocoo()
        elif m_f == SparseFormats.CSC:
            m: sp.csc_array = m.tocsc()

        if not self.skip_store:
            sp.save_npz(file_path, m)
        return m

    def load_ss(self, mtx: str, form: str, file: Path, ss: SuiteSparse) -> Union[sp.coo_array, sp.csr_array, sp.csc_array]:
        if form == "csr":
            mat: sp.csr_array = sp.load_npz(file)
            return mat

        # Always switch to a temporary directory for tar extraction
        with change_dir():
            extract_tar(file)
            nnz = ss.get_meta(mtx, "num_of_entries")
            self.args.dtype = "bool" if ss.is_binary(self.args.name) else "float64"
            self.rbio = RBio()
            mat: sp.csc_array = self.rbio.read_rb(Path(mtx) / (mtx + ".rb"), self.args.i, self.args.j, nnz, self.args.dtype)
            if form == "coo":
                mat: sp.coo_array = mat.tocoo(copy=False)
            return mat

    def get_ss_mat(self) -> Union[sp.csr_array, sp.coo_array, sp.csc_array]:
        ss = SuiteSparse(self.directory)
        mtx = self.args.name
        self.args.i = ss.get_meta(mtx, "num_of_rows")
        self.args.j = ss.get_meta(mtx, "num_of_cols")
        self.args.dtype = "bool" if ss.is_binary(self.args.name) else "float64"

        compressed = self.directory / f"{mtx}.npz" if self.args.matrix_format == "csr" \
            else self.directory / f"{mtx}.tar.gz"

        if not self.skip_load and compressed.exists():
            return self.load_ss(mtx, self.args.matrix_format, compressed, ss)

        # If skip_store is False, use a temporary dir to download
        download_dir = None if self.skip_store else self.directory
        with change_dir(download_dir):
            ss.get_matrix(self.args.name)
            file_path = Path(f"{mtx}.tar.gz")

            # If the requested format is csr, we need to load it as csc and then transform it to csr
            form = "csc" if self.args.matrix_format == "csr" else self.args.matrix_format
            mat = self.load_ss(mtx, form, file_path, ss)

            if self.args.matrix_format == "csr":
                mat = mat.tocsr(copy=False)
                if not self.skip_store:
                    out = mtx + ".npz"
                    sp.save_npz(out, mat.tocsr(copy=False))

            return mat

    def create_sparse_mat_and_dense_vec(self) -> Tuple[Union[sp.coo_array, sp.csr_array], np.ndarray]:
        return self.create_sparse_mat(), self.create_dense_vec()
