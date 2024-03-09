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
from typing import Dict, List, Tuple, Union

from logging_and_graphing import log_execution_times_secs
from utils import (create_sparse_mat_and_dense_vec, get_spmv_arg_parser,
                   make_work_dir_and_cd_to_it, run_func_and_capture_stdout)


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
        match = re.match(r"Thread (\d+) on core (-?\d+) (\w+)\((\d+)\) start ([\d\.]+) s end ([\d\.]+) s row (\d+) B_vals/crd\[(\d+):(\d+)\]", line)
        if match:
            thread_id, core_id, task_type, task_id, start_s, end_s, row, j_start, j_end = match.groups()
            start_time = float(start_s)
            end_time = float(end_s)
            all_start_times.append(start_time)
            tasks.append({"type": task_type,
                          "id": int(task_id),
                          "start": start_time,
                          "end": end_time,
                          "j_start": j_start,
                          "j_end": j_end,
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


def build_spmv(src: Path, pd: int, loc_hint: int, enable_logs: bool) -> Path:
    clang = Path(environ['LLVM_PATH']) / "bin/clang"
    assert clang.exists()

    generate = ["cmake", f"-DCMAKE_C_COMPILER={clang}", f"-DPREFETCH_DISTANCE={pd}", f"-DLOCALITY_HINT={loc_hint}", "-Bbuild", f"-S{src}"]
    if enable_logs:
        generate.append("-DENABLE_LOGS=y")
    run(generate, check=True)

    build = ["cmake", "--build", "build", "--target", "spmv-runahead"]
    run(build, check=True)

    lib_name = "libspmv-runahead" + (".dylib" if system() == "Darwin" else ".so")
    lib_path = Path("./build").resolve() / lib_name
    assert lib_path.exists(), f"Could not find {lib_name}"

    return lib_path


def parse_args() -> argparse.Namespace:
    common_arg_parser = get_spmv_arg_parser()

    parser = argparse.ArgumentParser(description="(Sparse Matrix)x(Dense Vector) Multiplication (SpMV) "
                                                 "with runahead prefetching using OpenMP")

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # profile
    profile_parser = subparsers.add_parser("profile", parents=[common_arg_parser],
                                           help="Profile the application.")

    # benchmark
    benchmark_parser = subparsers.add_parser("benchmark", parents=[common_arg_parser],
                                             help="Benchmark the application.")
    benchmark_parser.add_argument("--repetitions", type=int, default=5,
                                  help="Repeat the kernel with the same input. Gather execution times stats")

    # test
    test_parser = subparsers.add_parser("test", parents=[common_arg_parser],
                                        help="Generate logs and perform various tests based on the logs")
    test_parser.add_argument("--enable-logs", action="store_true", help="Enable logs")

    return parser.parse_args()


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


def check_same_iteration_range(pref: List[Dict], comp: List[Dict]):

    for i in range(len(pref)):
        assert pref[i]["j_start"] == comp[i]["j_start"] and pref[i]["j_end"] == comp[i]["j_end"], \
            f"pref task {i} and comp task {i} have different iteration ranges"


def check_task_dependencies_and_affinity(tasks: List[Dict]):

    prefs = [t for t in tasks if t["type"] == "pref"]
    prefs.sort(key=lambda x: x["id"])

    comps = [t for t in tasks if t["type"] == "comp"]
    comps.sort(key=lambda x: x["id"])

    assert len(prefs) == len(comps)

    check_sequential(prefs)
    check_sequential(comps)
    check_pref_i_depends_on_comp_i_minus_2(prefs, comps)
    check_same_iteration_range(prefs, comps)
    check_task_affinity(prefs, comps)


def benchmark(args: argparse.Namespace, shared_lib: Path, mat: Union[sp.coo_array, sp.csr_array], vec: np.ndarray):

    exec_times = []
    for i in range(args.epetitions):
        result, _, elapsed_wtime = run_spmv(shared_lib, mat, vec)
        exec_times.append(elapsed_wtime)

        expected = mat.dot(vec)
        assert np.allclose(result, expected), "Wrong result!"

        log_execution_times_secs(exec_times)


def test(args: argparse.Namespace, shared_lib: Path, mat: Union[sp.coo_array, sp.csr_array], vec: np.ndarray):
    result, stdout, elapsed_wtime = run_spmv(shared_lib, mat, vec)

    with open("stdout.txt", "a") as f:
        f.write(stdout)

    try:
        tasks = parse_logs(log=stdout.splitlines())
        check_task_dependencies_and_affinity(tasks)
        check_thread_affinity_and_spread(tasks)
    except AssertionError as e:
        print(f"Non critical check failed:\n {e}")

    expected = mat.dot(vec)
    assert np.allclose(result, expected), "Wrong result!"


def main():
    src_path = Path(__file__).parent.resolve() / "runahead-with-omp-tasks"
    assert src_path.exists()

    args = parse_args()

    mat, vec = create_sparse_mat_and_dense_vec(rows=args.rows, cols=args.cols, density=args.density, format="csr")

    make_work_dir_and_cd_to_it(__file__)

    if args.command == "test":
        shared_lib = build_spmv(src_path, args.prefetch_distance, args.locality_hint, enable_logs=True)
        test(args, shared_lib, mat, vec)
    elif args.command == "benchmark":
        shared_lib = build_spmv(src_path, args.prefetch_distance, args.locality_hint, enable_logs=False)
        benchmark(args, shared_lib, mat, vec)
    elif args.command == "profile":
        pass


if __name__ == "__main__":
    main()
