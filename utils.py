import argparse
import ctypes
import json
from multiprocessing import shared_memory
import numpy as np
from os import chdir, close, dup, dup2, environ, fsync, makedirs
from pathlib import Path
from platform import system
from shutil import rmtree, which
from subprocess import run
import sys
from tempfile import TemporaryFile
from typing import Any, Callable, List, Tuple

import scipy.sparse as sp


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


def read_config(file: str, key: str) -> List[str]:
    script_dir = Path(__file__).parent.resolve()
    cfg_file = script_dir / file

    cfg = []
    try:
        with open(cfg_file, "r") as f:
            print(f"Reading config {cfg_file}")
            cfg = json.load(f)[key]
    except FileNotFoundError:
        print(f"No {file} in {script_dir}")
    except json.decoder.JSONDecodeError as e:
        print(f"{cfg_file} could not be decoded {e}")
    except KeyError:
        print(f"{cfg_file} does not have a field for {key}")
    finally:
        return cfg


def get_spmv_arg_parser(with_pd: bool = True, with_loc_hint: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("-r", "--rows", type=int, default=1024,
                               help="Number of rows (default=1024)")
    parser.add_argument("-c", "--cols", type=int, default=1024,
                               help="Number of columns (default=1024)")
    parser.add_argument("-d", "--density", type=float, default=0.05,
                        help="Density of sparse matrix (default=0.05)")

    if with_pd:
        parser.add_argument("-pd", "--prefetch-distance", type=int, default=16,
                                   help="Prefetch distance")

    if with_loc_hint:
        parser.add_argument("-l", "--locality-hint", type=int, choices=[0, 1, 2, 3], default=3,
                            help="Temporal locality hint for prefetch instructions, "
                                 "3 for maximum temporal locality, 0 for no temporal locality. "
                                 "On x86, value 3 will produce PREFETCHT0, while value 0 will produce PREFETCHNTA")

    return parser


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


def build_with_cmake(cmake_args: List[str], target: str, src_path: Path, is_lib: bool = False) -> Path:

    # Prefer clang from user-specified llvm installation path
    clang = "clang"
    try:
        clang = Path(environ['LLVM_PATH']) / "bin/clang"
    except KeyError:
        pass

    assert is_in_path(clang)

    gen_cmd = ["cmake", f"-DCMAKE_C_COMPILER={clang}"] + cmake_args + ["-Bbuild", f"-S{src_path}"]
    run(gen_cmd, check=True)

    build_cmd = ["cmake", "--build", "build", "--target", target]
    run(build_cmd, check=True)

    artifact = target
    if is_lib:
        artifact = "lib" + target + (".dylib" if system() == "Darwin" else ".so")

    artifact_path = Path("./build").resolve() / artifact
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


def profile_spmv_with_vtune(exe: Path, mat: sp.csr_array, vec: np.ndarray, vtune_config: str):

    # copy vec to a shared mem block
    vec_shm = shared_memory.SharedMemory(create=True, size=vec.nbytes)
    shared_vec = np.ndarray(vec.shape, dtype=vec.dtype, buffer=vec_shm.buf)
    np.copyto(shared_vec, vec)

    # copy mat.data
    mat_data_shm = shared_memory.SharedMemory(create=True, size=mat.data.nbytes)
    shared_mat_data = np.ndarray(mat.data.shape, dtype=mat.data.dtype, buffer=mat_data_shm.buf)
    np.copyto(shared_mat_data, mat.data)

    # copy mat.indices
    mat_indices_shm = shared_memory.SharedMemory(create=True, size=mat.indices.nbytes)
    shared_mat_indices = np.ndarray(mat.indices.shape, dtype=mat.indices.dtype, buffer=mat_indices_shm.buf)
    np.copyto(shared_mat_indices, mat.indices)

    # copy mat.indptr
    mat_indptr_shm = shared_memory.SharedMemory(create=True, size=mat.indptr.nbytes)
    shared_mat_indptr = np.ndarray(mat.indptr.shape, dtype=mat.indptr.dtype, buffer=mat_indptr_shm.buf)
    np.copyto(shared_mat_indptr, mat.indptr)

    # create res buffer
    num_of_rows = mat.shape[0]
    all_zeroes = np.zeros(num_of_rows, dtype=mat.data.dtype)
    res_shm = shared_memory.SharedMemory(create=True, size=all_zeroes.nbytes)
    res = np.ndarray(all_zeroes.shape, dtype=mat.data.dtype, buffer=res_shm.buf)
    np.copyto(res, all_zeroes)

    try:
        if is_in_path("vtune"):
            vtune_cmd = ["vtune"] + read_config("vtune-config.json", vtune_config) + ["--"]
        else:
            vtune_cmd = []
            print("vtune not in PATH")

        spmv_cmd = [exe, str(num_of_rows), "/" + vec_shm.name, "/" + mat_data_shm.name, "/" + mat_indptr_shm.name,
                    "/" + mat_indices_shm.name, "/" + res_shm.name]
        run(vtune_cmd + spmv_cmd, check=True)
        expected = mat.dot(vec)
        assert np.allclose(res, expected), "Wrong result!"
    finally:
        vec_shm.unlink()
        mat_data_shm.unlink()
        mat_indptr_shm.unlink()
        mat_indices_shm.unlink()
        res_shm.unlink()


def run_spmv_as_foreign_fun(lib_path: Path, mat: sp.csr_array, vec: np.ndarray) -> Tuple[np.ndarray, str, float]:
    lib = ctypes.CDLL(str(lib_path))

    lib.compute.argtypes = [
        ctypes.c_uint64, # num_of_rows
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
