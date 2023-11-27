import argparse
import ctypes
import os
import platform
import signal
import subprocess
from shutil import rmtree
from os import chdir, environ, makedirs
from pathlib import Path
from time import sleep

from jinja2 import Template
import numpy as np
from scipy.sparse import random as sparse_random
from scipy.sparse import coo_matrix
from scipy.io import mmwrite

from mlir import runtime as rt
from mlir.execution_engine import *
from mlir import ir
from mlir.passmanager import *


def render_template(rows: int, cols: int, template_path: Path):

    with (open(template_path, "r") as template_f,
          open("spmv.mlir", "w") as rendered_template_f):
        rendered_template = Template(template_f.read()).render(rows=rows, cols=cols)
        rendered_template_f.write(rendered_template)


def lower_to_llvm() -> ir.Module:

    passes = ["lower-sparse-ops-to-foreach",
              "lower-sparse-foreach-to-scf",
              "sparsification",
              "sparse-reinterpret-map",
              "sparse-tensor-conversion",
              "canonicalize",
              "tensor-bufferize",
              "func-bufferize",
              "bufferization-bufferize",
              "convert-scf-to-cf",
              "convert-to-llvm"]

    def lower(p: str):
        lower.call_count += 1
        try:
            pm = PassManager.parse(f"builtin.module({p})")
        except ValueError:
            pm = PassManager.parse(f"builtin.module(func.func({p}))")

        pm.run(module.operation)
        with open(f"spmv.{lower.call_count}.{p}.mlir", "w") as f:
            f.write(str(module))

    with open("spmv.mlir", "r") as f:
        src = f.read()

    with ir.Context():
        module = ir.Module.parse(src)

        lower.call_count = 0
        for p in passes:
            lower(p)

    with open("spmv.llvm.mlir", "w") as f:
        f.write(str(module))

    return module


def create_sparse_mtx(rows: int, cols: int) -> np.ndarray:

    # create and store sparse matrix
    density = 0.05
    sparse_mat = sparse_random(rows, cols, density, dtype=np.float64).toarray()
    print(f"Sparse Mat val size: {(rows * cols * density * 8) / 1024**2} MB")
    return sparse_mat


@ctypes.CFUNCTYPE(ctypes.c_void_p)
def start_measurement_callback():
    print("Start Measurement Callback")

    # There's no perf on macos
    if platform.system() == "Darwin":
        return

    spmv_pid = os.getpid()
    start_measurement_callback.perf_proc = subprocess.Popen(["toplev", "-l4", "--raw", "--pid", f"{spmv_pid}"],
                                                            start_new_session=True)
    sleep(1)
    

@ctypes.CFUNCTYPE(ctypes.c_void_p)
def stop_measurement_callback():
    print("Stop Measurement Callback")

    # There's no perf on macos
    if platform.system() == "Darwin":
        return

    os.killpg(start_measurement_callback.perf_proc.pid, signal.SIGINT)
    start_measurement_callback.perf_proc.wait()


def run_spmv(llvm_mlir: ir.Module, rows: int, cols: int):

    llvm_path = environ.get("LLVM_PATH", None)
    if llvm_path is None:
        raise RuntimeError("Env var LLVM_PATH not specified")

    mlir_runtime = "libmlir_c_runner_utils"
    shared_lib_suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    mlir_runtime_path = Path(llvm_path) / "lib" / (mlir_runtime + shared_lib_suffix)

    mtx = create_sparse_mtx(rows, cols)
    mtx_mem = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(mtx)))

    dense_vec = np.array(cols * [1.0], np.float64)
    vec_mem = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(dense_vec)))

    res = np.zeros(rows, np.float64)
    res_mem = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(res)))

    # Allocate a MemRefDescriptor to receive the output tensor.
    # The buffer itself is allocated inside the MLIR code generation.
    ref_out = rt.make_nd_memref_descriptor(1, ctypes.c_double)()
    mem_out = ctypes.pointer(ctypes.pointer(ref_out))

    exec_engine = ExecutionEngine(llvm_mlir, shared_libs=[str(mlir_runtime_path)])
    exec_engine.register_runtime("start_measurement_callback", start_measurement_callback)
    exec_engine.register_runtime("stop_measurement_callback", stop_measurement_callback)

    exec_engine.invoke("main", mem_out, mtx_mem, vec_mem, res_mem)

    exec_engine.dump_to_object_file("spmv.o")

    # Sanity check on computed result.
    expected = np.matmul(mtx, dense_vec)
    c = rt.ranked_memref_to_numpy(mem_out[0])
    if np.allclose(c, expected):
        pass
    else:
        quit(f"FAILURE")


def main(rows: int, cols: int):

    build_path = Path("./build")

    if build_path.exists():
        rmtree(build_path)
    makedirs(build_path)

    template_path = Path("./spmv.mlir.jinja2").absolute()
    chdir(build_path)

    render_template(rows, cols, template_path)
    llvm_mlir = lower_to_llvm()

    run_spmv(llvm_mlir, rows, cols)


def get_args():
    parser = argparse.ArgumentParser(description="Process rows and cols.")
    parser.add_argument("--rows", type=int, default=1024, help="Number of rows (default=1024)")
    parser.add_argument("--cols", type=int, default=1024, help="Number of columns (default=1024)")

    args = parser.parse_args()
    return args.rows, args.cols


if __name__ == "__main__":
    rows, cols = get_args()

    print(f'vec size: {(cols * 8) / 1024} KB')

    main(rows, cols)
