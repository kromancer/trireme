import numpy as np
from os import chdir, close, dup, dup2, fsync, makedirs
from pathlib import Path
import scipy.sparse as sp
from shutil import rmtree
import sys
from tempfile import TemporaryFile
from typing import Any, Callable, Tuple, Union


def create_sparse_mat_and_dense_vec(rows: int, cols: int, density: float,
                                    format: str = "coo") -> Tuple[Union[sp.coo_array, sp.csr_array], np.ndarray]:

    dense_vec = np.array(cols * [1.0], np.float64)
    print(f'vector: size: {(cols * 8) / (1024 * 1024)} MB')

    rng = np.random.default_rng(5)
    sparse_mat = sp.random_array((rows, cols), density=density, dtype=np.float64, format=format, random_state=rng)
    print(f"sparse matrix: density: non-zero values size: {(rows * cols * density * 8) / 1024**2} MB, "
          f"density: {density}%")

    return sparse_mat, dense_vec


def make_work_dir_and_cd_to_it(file_path: str):
    build_path = Path(f"./workdir-{Path(file_path).name}")

    if build_path.exists():
        rmtree(build_path)
    makedirs(build_path)

    chdir(build_path)


def run_func_and_capture_stdout(func: Callable, *args, **kwargs) -> Tuple[str, Any]:
    # Backup the original stdout file descriptor
    stdout_fd = sys.stdout.fileno()
    original_stdout_fd = dup(stdout_fd)

    # Create a temporary file to capture the output
    with TemporaryFile(mode='w+b') as tmpfile:
        # Redirect stdout to the temporary file
        dup2(tmpfile.fileno(), stdout_fd)

        try:
            # Call the function
            res = func(*args, **kwargs)

            # Flush any buffered output
            fsync(stdout_fd)

            # Go back to the start of the temporary file to read its contents
            tmpfile.seek(0)
            captured = tmpfile.read().decode()
        finally:
            # Restore the original stdout file descriptor
            dup2(original_stdout_fd, stdout_fd)
            close(original_stdout_fd)

    return captured, res
