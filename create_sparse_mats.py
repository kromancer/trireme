from typing import Tuple, Union

import numpy as np
import scipy.sparse as sp

from utils import print_size


def create_sparse_mat_and_dense_vec(rows: int, cols: int, density: float, form: str = "coo") -> Tuple[Union[sp.coo_array, sp.csr_array], np.ndarray]:

    rng = np.random.default_rng(5)

    dense_vec = rng.random(cols)
    print(f"vector: size: {print_size(cols * np.float64().itemsize)}")

    sparse_mat = sp.random_array((rows, cols), density=density, dtype=np.float64, format=form, random_state=rng)
    print(f"sparse matrix: non-zero values size: {print_size(rows * cols * density * np.float64().itemsize)}, "
          f"density: {density}%")

    return sparse_mat, dense_vec
