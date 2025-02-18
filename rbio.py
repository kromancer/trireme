import ctypes
from pathlib import Path
import platform
from typing import Union
import numpy as np
import scipy.sparse as sp

from common import timeit


class RBio:
    def __init__(self):
        self.lib = self._load_lib()
        self.malloced = []
        self.suite_sparse_free = None
        self.suite_sparse_start = None

        # Initialize SuiteSparse
        if self.lib:
            self._initialize_suite_sparse()

    @staticmethod
    def _load_lib():
        # Determine the correct library name based on the operating system
        if platform.system() == 'Darwin':
            lib_name = 'librbio.dylib'
            env_var = 'DYLD_LIBRARY_PATH'
        elif platform.system() == 'Linux':
            lib_name = 'librbio.so'
            env_var = 'LD_LIBRARY_PATH'
        else:
            assert False, "Unsupported OS"

        try:
            # Attempt to load the library
            return ctypes.CDLL(lib_name)
        except OSError as e:
            # Handle failure to load the library
            print(e)
            print(f"Failed to load {lib_name}.")
            print(f"Make sure the library exists in your system and the directory containing it is added to {env_var}.")
            exit(1)

    def _initialize_suite_sparse(self):
        suite_sparse_start = self.lib.SuiteSparse_start
        suite_sparse_start.restype = None
        suite_sparse_start.argtypes = []
        suite_sparse_start()

        self.suite_sparse_free = self.lib.SuiteSparse_free
        self.suite_sparse_free.argtypes = [ctypes.c_void_p]
        self.suite_sparse_free.restype = ctypes.c_void_p

    def _read(self, mtx: Path, index_type: Union[type[ctypes.c_int64], type[ctypes.c_int64]], dtype: str) -> sp.csc_array:
        # RBread
        if index_type == ctypes.c_int64:
            rbread = self.lib.RBread
        elif index_type == ctypes.c_int32:
            rbread = self.lib.RBread_i
        else:
            assert False, "Unsupported index type"

        rbread.restype = ctypes.c_int
        rbread.argtypes = [
            ctypes.c_char_p,  # filename
            index_type,   # build_upper
            index_type,   # zero_handling
            ctypes.c_char_p,  # title
            ctypes.c_char_p,  # key
            ctypes.c_char_p,  # mtype
            ctypes.POINTER(index_type),  # nrow
            ctypes.POINTER(index_type),  # ncol
            ctypes.POINTER(index_type),  # mkind
            ctypes.POINTER(index_type),  # skind
            ctypes.POINTER(index_type),  # asize
            ctypes.POINTER(index_type),  # znz
            ctypes.POINTER(ctypes.POINTER(index_type)),  # p_Ap
            ctypes.POINTER(ctypes.POINTER(index_type)),  # p_Ai
            ctypes.c_void_p,  # p_Ax -> all below will be set to null
            ctypes.c_void_p,  # p_Az
            ctypes.c_void_p,  # p_Zp
            ctypes.c_void_p,  # p_Zi
        ]

        # Prepare inputs
        filename = ctypes.create_string_buffer(str(mtx).encode('ascii'))
        title = ctypes.create_string_buffer(73)
        key = ctypes.create_string_buffer(9)
        mtype = ctypes.create_string_buffer(4)
        nrow = index_type()
        ncol = index_type()
        mkind = index_type()
        skind = index_type()
        asize = index_type()
        znz = index_type()
        p_Ap = ctypes.POINTER(index_type)()
        p_Ai = ctypes.POINTER(index_type)()
        null_ptr = ctypes.c_void_p()

        result = rbread(
            filename,
            index_type(0),  # No special handling of symmetric matrices
            index_type(0),  # Do not prune explicitly stored zeroes
            title,
            key,
            mtype,
            ctypes.byref(nrow),
            ctypes.byref(ncol),
            ctypes.byref(mkind),
            ctypes.byref(skind),
            ctypes.byref(asize),
            ctypes.byref(znz),
            ctypes.byref(p_Ap),
            ctypes.byref(p_Ai),
            null_ptr,  # NULL pointer for p_Ax - read only shape
            null_ptr,  # NULL pointer for p_Az - we do not use complex arrays
            null_ptr,  # NULL pointer for p_Ax - we do not ask for zero pruning
            null_ptr,  # NULL pointer for p_Ax - we do not ask for zero pruning
        )
        assert result == 0, "RBread() failed"
        assert mkind.value == 1, "When reading shape, mkind should be set to pattern"

        self.p_Ap = p_Ap
        indptr = np.ctypeslib.as_array(p_Ap, shape=(ncol.value + 1,))
        self.p_Ai = p_Ai
        indices = np.ctypeslib.as_array(p_Ai, shape=(asize.value,))
        data = np.ones(indices.size, dtype=dtype)

        # TODO: Avoid this copy
        mat = sp.csc_array((data, indices, indptr), copy=True, shape=(nrow.value, ncol.value))

        self.free_and_finish()
        return mat

    @timeit
    def read_rb(self, mtx: Path, i: int, j: int, nnz: int, dtype: str) -> sp.csc_array:
        if all(n < 2 ** 31 for n in [i, j, nnz]):
            return self._read(mtx, ctypes.c_int32, dtype)
        else:
            return self._read(mtx, ctypes.c_int64, dtype)

    def free_and_finish(self):
        suite_sparse_finish = self.lib.SuiteSparse_finish
        suite_sparse_finish.restype = None
        suite_sparse_finish.argtypes = []
        self.suite_sparse_free(self.p_Ap)
        self.suite_sparse_free(self.p_Ai)
        suite_sparse_finish()
