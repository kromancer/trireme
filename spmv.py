import argparse
import ctypes
import os
import platform
import signal
import statistics
import subprocess
from shutil import rmtree
from os import chdir, environ, makedirs
from pathlib import Path
from time import sleep
from typing import List, Optional, Tuple

from jinja2 import Template
import numpy as np
import scipy.sparse as sp

from mlir import runtime as rt
from mlir.execution_engine import *
from mlir import ir
from mlir.passmanager import *


remaining_repetitions = 1
execution_times = []


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


def create_sparse_mtx_and_dense_vec(rows: int, cols: int, density: float) -> Tuple[np.ndarray, np.ndarray]:

    dense_vec = np.array(cols * [1.0], np.float64)
    print(f'vector: size: {(cols * 8) / (1024 * 1024)} MB')

    rng = np.random.default_rng(5)
    sparse_mat = sp.random_array((rows, cols), density=density, dtype=np.float64, random_state=rng).toarray()
    print(f"sparse matrix: density: non-zero values size: {(rows * cols * density * 8) / 1024**2} MB, "
          f"density: {density}%")

    return sparse_mat, dense_vec


@ctypes.CFUNCTYPE(ctypes.c_void_p)
def start_measurement_callback():
    print("kernel start")

    # There's no perf on macos
    if platform.system() == "Darwin":
        return

    # Run perf only for the last repetition
    if remaining_repetitions > 1:
        return

    spmv_pid = os.getpid()
    start_measurement_callback.perf_proc = subprocess.Popen(["toplev", "-l6", "--pid", f"{spmv_pid}"],
                                                            start_new_session=True)
    sleep(1)


@ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint64)
def stop_measurement_callback(dur_ns: int):
    global remaining_repetitions, execution_times

    dur_ms = round(dur_ns/1000000, 3)
    print(f"kernel finish: execution time: {dur_ms} ms")
    execution_times.append(dur_ns)

    # There's no perf on macos
    if platform.system() == "Darwin":
        return

    if remaining_repetitions > 1:
        remaining_repetitions -= 1
        return

    os.killpg(start_measurement_callback.perf_proc.pid, signal.SIGINT)
    start_measurement_callback.perf_proc.wait()


def run_spmv(llvm_mlir: ir.Module, rows: int, mtx: np.ndarray, vec: np.ndarray):

    llvm_path = environ.get("LLVM_PATH", None)
    if llvm_path is None:
        raise RuntimeError("Env var LLVM_PATH not specified")

    runtimes = ["libmlir_runner_utils", "libmlir_c_runner_utils"]
    shared_lib_suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    runtime_paths = [str(Path(llvm_path) / "lib" / (r + shared_lib_suffix)) for r in runtimes]

    mtx_mem = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(mtx)))
    vec_mem = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(vec)))

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
    expected = np.matmul(mtx, vec)
    c = rt.ranked_memref_to_numpy(mem_out[0])
    assert np.allclose(c, expected), "Wrong output!"


def make_build_dir_and_cd_to_it(file_path: str):
    build_path = Path(f"./build-{Path(file_path).name}")

    if build_path.exists():
        rmtree(build_path)
    makedirs(build_path)

    chdir(build_path)


def filtered_by_median(data) -> List[int]:
    r""" Temporary hack until we can guarantee that there is no context switching to the core running the kernel"""
    median_value = np.median(data)

    filtered = [x for x in data if x <= 1.5 * median_value]
    outliers = [x for x in data if x not in filtered]

    if outliers:
        print("Outliers:", outliers)

    return filtered


def main():
    global remaining_repetitions

    rows, cols, density, template_path, remaining_repetitions, passes = get_args()

    make_build_dir_and_cd_to_it(__file__)

    src = render_template(rows, cols, template_path)

    llvm_mlir = apply_passes(src, passes)

    mtx, vec = create_sparse_mtx_and_dense_vec(rows, cols, density)

    repetitions = remaining_repetitions
    for _ in range(0, repetitions):
        run_spmv(llvm_mlir=llvm_mlir, rows=rows, mtx=mtx, vec=vec)

    if repetitions > 1:
        filtered = filtered_by_median(execution_times)
        mean = round(statistics.mean(filtered) / 1000000, 3)
        std_dev = round(statistics.stdev(filtered) / 1000000, 3)
        cv = round(std_dev / mean, 3)
        print(f"mean execution time: {mean} ms")
        print(f"std dev: {std_dev} ms, CV: {cv} %")


def get_template_path(opt: Optional[str]) -> Path:
    return Path({
        "pref-ains": "./spmv.prefetch.mlir.jinja2",
        "pref-spe": "./spmv.spe.mlir.jinja2"
    }.get(opt, "./spmv.mlir.jinja2")).absolute()


def get_mlir_opt_passes(opt: Optional[str]) -> List[str]:
    if opt == "vect-vl4":
        return ["lower-sparse-ops-to-foreach",
                "lower-sparse-foreach-to-scf",
                "sparse-reinterpret-map",
                "sparse-vectorization{vl=4}",
                "sparse-tensor-conversion",
                "tensor-bufferize",
                "func-bufferize",
                "bufferization-bufferize",
                "convert-scf-to-cf",
                f"convert-vector-to-llvm{{{'enable-x86vector' if platform.machine() == 'x86_64' else 'enable-arm-neon'}}}",
                "lower-affine",
                "convert-arith-to-llvm",
                "convert-to-llvm",
                "reconcile-unrealized-casts"]
    else:
        return ["lower-sparse-ops-to-foreach",
                "lower-sparse-foreach-to-scf",
                "sparse-reinterpret-map",
                "sparse-tensor-conversion",
                "canonicalize",
                "tensor-bufferize",
                "func-bufferize",
                "bufferization-bufferize",
                "convert-scf-to-cf",
                "convert-to-llvm"]


def get_args():
    parser = argparse.ArgumentParser(description="Process rows and cols.")
    parser.add_argument("-r", "--rows", type=int, default=1024,
                        help="Number of rows (default=1024)")
    parser.add_argument("-c", "--cols", type=int, default=1024,
                        help="Number of columns (default=1024)")
    parser.add_argument("-d", "--density", type=float, default=0.05,
                        help="Density of sparse matrix (default=0.05)")
    parser.add_argument("-o", "--optimization", choices=["vect-vl4", "pref-ains", "pref-spe"],
                        help="Use an optimized version of the kernel")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="Repeat the kernel with the same input. "
                        "Gather execution times, only run perf for the last run")

    args = parser.parse_args()
    return (args.rows, args.cols, args.density, get_template_path(args.optimization), args.repetitions,
            get_mlir_opt_passes(args.optimization))


if __name__ == "__main__":
    main()
