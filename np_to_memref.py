import numpy as np
import ctypes


def make_nd_memref_descriptor(rank: int, dtype: np.dtype, itype: np.dtype):

    dt = np.ctypeslib.as_ctypes_type(dtype)
    it = np.ctypeslib.as_ctypes_type(itype)

    class MemRefDescriptor(ctypes.Structure):
        """Builds an empty descriptor for the given rank/dtype, where rank>0."""

        _fields_ = [
            ("allocated", ctypes.c_longlong),
            ("aligned", ctypes.POINTER(dt)),
            ("offset", it),
            ("shape", it * rank),
            ("strides", it * rank),
        ]

    return MemRefDescriptor
