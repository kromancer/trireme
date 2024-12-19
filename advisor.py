from argparse import Namespace
from multiprocessing import shared_memory
from os import environ
from pathlib import Path
from platform import machine
from subprocess import run

from common import is_in_path, read_config
import numpy as np
import scipy.sparse as sp


def compile_exe(spmv_ll: Path):
    clang = Path(environ['LLVM_PATH']) / "bin/clang"
    assert clang.exists()

    compile_spmv_cmd = [str(clang), "-O3", "-mavx2" if machine() == 'x86_64' else "",
                        "-Wno-override-module", "-c", str(spmv_ll), "-o", "spmv.o"]
    run(compile_spmv_cmd, check=True)

    main_fun = Path(__file__).parent.resolve() / "templates" / "spmv_csr.main.c"
    compile_exe_cmd = [str(clang), "-O0", "-fopenmp", "-g", str(main_fun), "spmv.o", "-o" "spmv"]
    run(compile_exe_cmd, check=True)


def profile_spmv_with_advisor(args: Namespace, spmv_ll: Path, mat: sp.csr_array, vec: np.ndarray):
    assert is_in_path("advisor")

    compile_exe(spmv_ll)

    # copy vec to a shared mem block
    cols = vec.shape[0]
    vec_shm = shared_memory.SharedMemory(create=True, size=vec.nbytes)
    shared_vec = np.ndarray(vec.shape, dtype=vec.dtype, buffer=vec_shm.buf)
    np.copyto(shared_vec, vec)

    # copy mat.data
    nnz = mat.data.shape[0]
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
    rows = mat.shape[0]
    all_zeroes = np.zeros(rows, dtype=mat.data.dtype)
    res_shm = shared_memory.SharedMemory(create=True, size=all_zeroes.nbytes)
    res = np.ndarray(all_zeroes.shape, dtype=mat.data.dtype, buffer=res_shm.buf)
    np.copyto(res, all_zeroes)

    spmv_cmd = ["./spmv",
                str(rows),
                str(cols),
                str(nnz),
                "/" + vec_shm.name,
                "/" + mat_data_shm.name,
                "/" + mat_indptr_shm.name,
                "/" + mat_indices_shm.name,
                "/" + res_shm.name]

    cmd = ["advisor"] + read_config("advisor-config.json", args.config) + ["--"] + spmv_cmd

    try:
        run(cmd, check=True)

        if args.check_output:
            expected = mat.dot(vec)
            assert np.allclose(res, expected), "Wrong output!"
    finally:
        vec_shm.unlink()
        mat_data_shm.unlink()
        mat_indptr_shm.unlink()
        mat_indices_shm.unlink()
        res_shm.unlink()
