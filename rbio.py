import ctypes
from pathlib import Path
import platform


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
            print("Unsupported OS")
            return

        try:
            # Attempt to load the library
            return ctypes.CDLL(lib_name)
        except OSError as e:
            # Handle failure to load the library
            print(f"Failed to load {lib_name}.")
            print(f"Make sure the library exists in your system and the directory containing it is added to {env_var}.")

    def _initialize_suite_sparse(self):
        suite_sparse_start = self.lib.SuiteSparse_start
        suite_sparse_start.restype = None
        suite_sparse_start.argtypes = []
        suite_sparse_start()

        self.suite_sparse_free = self.lib.SuiteSparse_free
        self.suite_sparse_free.argtypes = [ctypes.c_void_p]
        self.suite_sparse_free.restype = None

    def read_rb(self, mtx: Path):
        # RBread
        rbread = self.lib.RBread
        rbread.restype = ctypes.c_int
        rbread.argtypes = [
            ctypes.c_char_p,  # filename
            ctypes.c_int64,   # build_upper
            ctypes.c_int64,   # zero_handling
            ctypes.c_char_p,  # title
            ctypes.c_char_p,  # key
            ctypes.c_char_p,  # mtype
            ctypes.POINTER(ctypes.c_int64),  # nrow
            ctypes.POINTER(ctypes.c_int64),  # ncol
            ctypes.POINTER(ctypes.c_int64),  # mkind
            ctypes.POINTER(ctypes.c_int64),  # skind
            ctypes.POINTER(ctypes.c_int64),  # asize
            ctypes.POINTER(ctypes.c_int64),  # znz
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),  # p_Ap
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),  # p_Ai
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
        nrow = ctypes.c_int64()
        ncol = ctypes.c_int64()
        mkind = ctypes.c_int64()
        skind = ctypes.c_int64()
        asize = ctypes.c_int64()
        znz = ctypes.c_int64()
        p_Ap = ctypes.POINTER(ctypes.c_int64)()
        p_Ai = ctypes.POINTER(ctypes.c_int64)()
        null_ptr = ctypes.c_void_p()

        result = rbread(
            filename,
            ctypes.c_int64(0),  # No special handling of symmetric matrices
            ctypes.c_int64(0),  # Do not prune explicitly stored zeroes
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

        self.malloced = [p_Ap, p_Ai]

        return nrow.value, ncol.value, asize.value, p_Ap, p_Ai

    def free_and_finish(self):
        suite_sparse_finish = self.lib.SuiteSparse_finish
        suite_sparse_finish.restype = None
        suite_sparse_finish.argtypes = []

        for p in self.malloced:
            self.suite_sparse_free(p)

        suite_sparse_finish()
