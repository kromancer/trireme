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
from typing import List

from jinja2 import Template
import numpy as np
from scipy.sparse import random as sparse_random

from mlir import runtime as rt
from mlir.execution_engine import *
from mlir import ir
from mlir.passmanager import *


def render_template(rows: int, cols: int, template_path: Path) -> str:

    with (open(template_path, "r") as template_f,
          open("spmv.mlir", "w") as rendered_template_f):
        rendered_template = Template(template_f.read()).render(rows=rows, cols=cols)
        rendered_template_f.write(rendered_template)

    return rendered_template


def apply_passes(src: str, passes: List[str]) -> ir.Module:

    def run_pass(p: str):
        run_pass.call_count += 1
        try:
            pm = PassManager.parse(f"builtin.module({p})")
        except ValueError:
            pm = PassManager.parse(f"builtin.module(func.func({p}))")

        pm.run(module.operation)
        with open(f"spmv.{run_pass.call_count}.{p}.mlir", "w") as f:
            f.write(str(module))

    with ir.Context():
        module = ir.Module.parse(src)

        run_pass.call_count = 0
        for p in passes:
            run_pass(p)

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
    print("\nStop Measurement Callback")

    # There's no perf on macos
    if platform.system() == "Darwin":
        return

    os.killpg(start_measurement_callback.perf_proc.pid, signal.SIGINT)
    start_measurement_callback.perf_proc.wait()


def run_spmv(llvm_mlir: ir.Module, rows: int, cols: int):

    llvm_path = environ.get("LLVM_PATH", None)
    if llvm_path is None:
        raise RuntimeError("Env var LLVM_PATH not specified")

    runtimes = ["libmlir_runner_utils", "libmlir_c_runner_utils"]
    shared_lib_suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    runtime_paths = [str(Path(llvm_path) / "lib" / (r + shared_lib_suffix)) for r in runtimes]

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

    exec_engine = ExecutionEngine(llvm_mlir, shared_libs=runtime_paths)
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


def make_build_dir_and_cd_to_it(file_path: str):
    build_path = Path(f"./build-{Path(file_path).name}")

    if build_path.exists():
        rmtree(build_path)
    makedirs(build_path)

    chdir(build_path)


def main(rows: int, cols: int, pref: bool):

    template_path = (Path("./spmv.prefetch.mlir.jinja2") if pref else Path("./spmv.mlir.jinja2")).absolute()

    make_build_dir_and_cd_to_it(__file__)

    src = render_template(rows, cols, template_path)

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

    llvm_mlir = apply_passes(src, passes)

    run_spmv(llvm_mlir, rows, cols)


def get_args():
    parser = argparse.ArgumentParser(description="Process rows and cols.")
    parser.add_argument("--rows", type=int, default=1024, help="Number of rows (default=1024)")
    parser.add_argument("--cols", type=int, default=1024, help="Number of columns (default=1024)")
    parser.add_argument("-p", "--prefetch", action="store_true", help="Enable prefetching.")

    args = parser.parse_args()
    return args.rows, args.cols, args.prefetch


if __name__ == "__main__":
    rows, cols, is_pref = get_args()

    print(f'vec size: {(cols * 8) / 1024} KB')

    main(rows, cols, is_pref)
