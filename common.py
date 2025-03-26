import argparse
from contextlib import contextmanager
import ctypes
from enum import Enum
from functools import wraps
import json
import numpy as np
from os import chdir, close, dup, dup2, environ, fsync, makedirs
from pathlib import Path
from platform import system
from shutil import rmtree, which
from subprocess import run
import sys
import tarfile
from tempfile import TemporaryDirectory, TemporaryFile
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import scipy.sparse as sp


class SparseFormats(Enum):
    CSR = "csr"
    COO = "coo"
    CSC = "csc"

    def __str__(self):
        return self.value


def print_size(size):
    KB = 1024
    MB = KB ** 2
    GB = KB ** 3

    if size >= GB:
        return f"{size / GB:.2f} GB"
    elif size >= MB:
        return f"{size / MB:.2f} MB"
    elif size >= KB:
        return f"{size / KB:.2f} KB"
    else:
        return f"{size} Bytes"


def read_config(file: str, key: str) -> Optional[Union[int, str, List[str], Dict]]:
    script_dir = Path(__file__).parent.resolve()
    cfg_file = script_dir / file

    cfg = None
    try:
        with open(cfg_file, "r") as f:
            cfg = json.load(f)[key]
    except FileNotFoundError:
        print(f"No {file} in {script_dir}")
    except json.decoder.JSONDecodeError as e:
        print(f"{cfg_file} could not be decoded {e}")
    except KeyError:
        print(f"{cfg_file} does not have a field for {key}")
    finally:
        return cfg


def timeit(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"Time taken by {func.__name__}: {end - start:.3f} seconds")
        return result
    return wrapper


def make_work_dir_and_cd_to_it(file_path: str):
    build_path = Path(f"./workdir-{Path(file_path).name}")

    if build_path.exists():
        rmtree(build_path)
    makedirs(build_path)

    chdir(build_path)


def is_in_path(exe: str) -> bool:

    exe_path = which(exe)

    if exe_path is None:
        print(f"{exe} not in PATH")
        return False

    return True


@timeit
def extract_tar(file: Path):
    if is_in_path("tar"):
        run(["tar", "-xf", file], check=True)
    else:
        with tarfile.open(file, "r:gz") as tar:
            tar.extractall()


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

        finally:
            # Flush any buffered output
            fsync(stdout_fd)

            # Go back to the start of the temporary file to read its contents
            tmpfile.seek(0)
            captured = tmpfile.read().decode()

            # Restore the original stdout file descriptor
            dup2(original_stdout_fd, stdout_fd)
            close(original_stdout_fd)

    return captured, res


def build_with_cmake(cmake_args: List[str], target: str, src_path: Path, is_lib: bool = False) -> Path:

    # Prefer clang from user-specified llvm installation path
    clang = "clang"
    try:
        clang = Path(environ['LLVM_PATH']) / "bin/clang"
    except KeyError:
        pass

    assert is_in_path(clang)

    # Use a build directory named after the target
    build_dir = "build-" + target

    gen_cmd = ["cmake", f"-DCMAKE_C_COMPILER={clang}"] + cmake_args + [f"-B{build_dir}", f"-S{src_path}"]
    run(gen_cmd, check=True)

    build_cmd = ["cmake", "--build", build_dir, "--target", target]
    run(build_cmd, check=True)

    artifact = target
    if is_lib:
        artifact = "lib" + target + (".dylib" if system() == "Darwin" else ".so")

    artifact_path = Path(build_dir).resolve() / artifact
    assert artifact_path.exists(), f"Could not find {artifact}"

    return artifact_path


def benchmark_spmv(args: argparse.Namespace, shared_lib: Path, mat: sp.csr_array, vec: np.ndarray) -> List[float]:
    exec_times = []
    for i in range(args.repetitions):
        result, _, elapsed_wtime = run_spmv_as_foreign_fun(shared_lib, mat, vec)
        exec_times.append(elapsed_wtime)

        expected = mat.dot(vec)
        assert np.allclose(result, expected), "Wrong result!"

    return exec_times


def run_spmv_as_foreign_fun(lib_path: Path, mat: sp.csr_array, vec: np.ndarray) -> Tuple[np.ndarray, str, float]:
    lib = ctypes.CDLL(str(lib_path))

    lib.compute.argtypes = [
        ctypes.c_uint64,  # num_of_rows
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # const double* vec_
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # const double* mat_vals_
        np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),    # const int64_t* pos
        np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),    # const int64_t* crd_
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # double *res_
    ]

    lib.compute.restype = ctypes.c_double

    num_of_rows = mat.shape[0]
    result_buff = np.array([0] * num_of_rows, dtype=np.float64)

    stdout, elapsed_wtime = run_func_and_capture_stdout(lib.compute, num_of_rows, vec, mat.data,
                                                        mat.indptr, mat.indices, result_buff)

    return result_buff, stdout, elapsed_wtime


@contextmanager
def change_dir(destination: Path = None):
    current_dir = Path.cwd()
    temp_dir = None

    try:
        if destination is None:
            temp_dir = TemporaryDirectory()
            destination = Path(temp_dir.name)
        chdir(destination)
        yield
    finally:
        chdir(current_dir)
        if temp_dir is not None:
            temp_dir.cleanup()


def flush_cache(cache_size=100 * 1024 * 1024):
    a = np.arange(cache_size, dtype=np.uint8)
    a.sum()  # forces reading the array
