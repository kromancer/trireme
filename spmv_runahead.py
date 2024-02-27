import argparse
import ctypes
import json
import numpy as np
from os import environ
from pathlib import Path
import re
from platform import system
import scipy.sparse as sp
from subprocess import run
from typing import Dict, List, Tuple

from logging_and_graphing import log_execution_times_secs
from utils import create_sparse_mat_and_dense_vec, make_work_dir_and_cd_to_it, run_func_and_capture_stdout


def run_spmv(lib_path: Path, mat: sp.csr_array, vec: np.ndarray) -> Tuple[np.ndarray, str, float]:
    lib = ctypes.CDLL(str(lib_path))

    lib.compute.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # double* a_vals_
        ctypes.c_int,  # int num_of_rows
        np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),  # const int* pos
        np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),  # const int* crd_
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # const double* B_vals_
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # const double* c_vals_
    ]

    lib.compute.restype = ctypes.c_double

    num_of_rows = mat.shape[0]
    result_buff = np.array([0] * num_of_rows, dtype=np.float64)

    stdout, elapsed_wtime = run_func_and_capture_stdout(lib.compute, result_buff, num_of_rows,
                                                        mat.indptr, mat.indices, mat.data, vec)

    return result_buff, stdout, elapsed_wtime


def parse_logs(log) -> List[Dict]:
    tasks = []
    all_start_times = []

    for line in log:
        match = re.match(r"Thread (\d+) on core (-?\d+) (\w+)\((\d+)\) start ([\d\.]+) s end ([\d\.]+) s", line)
        if match:
            thread_id, core_id, task_type, task_id, start_s, end_s = match.groups()
            start_time = float(start_s)
            end_time = float(end_s)
            all_start_times.append(start_time)
            tasks.append({"type": task_type,
                          "id": int(task_id),
                          "start": start_time,
                          "end": end_time,
                          "thread": int(thread_id),
                          "core": int(core_id)})

    # Time 0 is relative the start of the first event
    ref = min(all_start_times)
    for task in tasks:
        task["start"] -= ref
        task["end"] -= ref

    # Save tasks in a .json file
    with open("tasks.json", "w") as f:
        json.dump(tasks, f, indent=4)

    return tasks


# Check that x(i) depends on x(i - 1)
def check_sequential(tasks: List):
    for i in range(1, len(tasks)):
        assert tasks[i]["start"] >= tasks[i - 1]["end"], f"task {i} breaks sequential dependency on task {i - 1}"


# Check that pref(i) depends on comp(i - 2)
def check_pref_i_depends_on_comp_i_minus_2(pref_tasks: List, comp_tasks: List):
    for i in range(2, len(pref_tasks)):
        assert pref_tasks[i]["start"] >= comp_tasks[i - 2]["end"], \
            f"pref task {i} breaks dependency on comp task {i - 2}"


def build_spmv(src: Path, pd: int, enable_logs: bool) -> Path:
    clang = Path(environ['LLVM_PATH']) / "bin/clang"
    assert clang.exists()

    generate = ["cmake", f"-DCMAKE_C_COMPILER={clang}", f"-DPREFETCH_DISTANCE={pd}", "-Bbuild", f"-S{src}"]
    if enable_logs:
        generate.append("-DENABLE_LOGS=y")
    run(generate, check=True)

    build = ["cmake", "--build", "build"]
    run(build, check=True)

    lib_name = "libspmv-runahead" + (".dylib" if system() == "Darwin" else ".so")
    lib_path = Path("./build").resolve() / lib_name
    assert lib_path.exists(), f"Could not find {lib_name}"

    return lib_path


def parse_args() -> Tuple[int, int, int, float, bool, int]:
    parser = argparse.ArgumentParser(description="(Sparse Matrix)x(Dense Vector) Multiplication (SpMV) "
                                                 "with runahead prefetching using OpenMP")
    parser.add_argument("-r", "--rows", type=int, default=1024,
                        help="Number of rows (default=1024)")
    parser.add_argument("-c", "--cols", type=int, default=1024,
                        help="Number of columns (default=1024)")
    parser.add_argument("-pd", "--prefetch-distance", type=int, default=16,
                        help="Prefetch distance")
    parser.add_argument("-d", "--density", type=float, default=0.05,
                        help="Density of sparse matrix (default=0.05)")
    parser.add_argument("-l", "--enable-logs", action="store_true", default=False,
                        help="Enable logs")
    parser.add_argument("--repetitions", type=int, default=5,
                        help="Repeat the kernel with the same input. "
                        "Gather execution times, only run perf for the last run")

    args = parser.parse_args()

    return args.rows, args.cols, args.prefetch_distance, args.density, args.enable_logs, args.repetitions


def check_task_affinity(prefs: List[Dict], comps: List[Dict]):
    for i in range(0, len(prefs)):
        assert prefs[i]["core"] == comps[i]["core"], f"Task affinity mismatch for pref/comp ({comps[i]['id']})"


def check_thread_affinity_and_spread(tasks: List[Dict]):
    thread_ids = {t["thread"] for t in tasks}

    used_cores = []
    for thread_id in thread_ids:
        thread_execution_cores = {t["core"] for t in tasks if t["thread"] == thread_id}
        assert len(thread_execution_cores) == 1, (f"Thread affinity has been violated, thread {thread_id} "
                                                  f"executed on cores {thread_execution_cores}")
        thread_core = thread_execution_cores.pop()
        assert thread_core not in used_cores, f"Thread {thread_id} was bound to a core used by another thread"
        used_cores.append(thread_core)


def check_task_dependencies_and_affinity(tasks: List[Dict]):

    prefs = [t for t in tasks if t["type"] == "pref"]
    prefs.sort(key=lambda x: x["id"])

    comps = [t for t in tasks if t["type"] == "comp"]
    comps.sort(key=lambda x: x["id"])

    assert len(prefs) == len(comps)

    check_sequential(prefs)
    check_sequential(comps)
    check_pref_i_depends_on_comp_i_minus_2(prefs, comps)
    check_task_affinity(prefs, comps)


def main():
    src_path = Path(__file__).parent.resolve() / "runahead-with-omp-tasks"
    assert src_path.exists()

    rows, cols, pd, dens, enable_logs, repetitions = parse_args()

    make_work_dir_and_cd_to_it(__file__)
    shared_lib = build_spmv(src_path, pd, enable_logs)

    mat, vec = create_sparse_mat_and_dense_vec(rows=rows, cols=cols, density=dens, format="csr")

    wtimes = []
    for i in range(repetitions):
        result, stdout, elapsed_wtime = run_spmv(shared_lib, mat, vec)
        wtimes.append(elapsed_wtime)

        with open("stdout.txt", "a") as f:
            f.write(stdout)

        if enable_logs:
            try:
                tasks = parse_logs(log=stdout.splitlines())
                check_thread_affinity_and_spread(tasks)
                check_task_dependencies_and_affinity(tasks)
            except AssertionError as e:
                print(f"Non critical check failed:\n {e}")

        expected = mat.dot(vec)
        assert np.allclose(result, expected), "Wrong result!"

    log_execution_times_secs(wtimes)


if __name__ == "__main__":
    main()
