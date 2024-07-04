import argparse
import json
import numpy as np
from os import environ
from pathlib import Path
from platform import system
import re
import scipy.sparse as sp
from subprocess import run
from typing import Dict, List

from matrix_storage_manager import create_sparse_mat_and_dense_vec
from logging_and_graphing import log_execution_times_secs
from benchmark import add_parser_for_benchmark
from common import (add_parser_for_profile, benchmark_spmv, get_spmv_arg_parser, make_work_dir_and_cd_to_it,
                    run_spmv_as_foreign_fun)
from vtune import profile_spmv_with_vtune


def parse_logs(log) -> List[Dict]:
    tasks = []
    all_start_times = []

    for line in log:
        match = re.match(r"Thread (\d+) on core (-?\d+) (\w+)\((\d+)\) start ([\d\.]+) s end ([\d\.]+) s row (\d+) mat_vals/crd\[(\d+):(\d+)\]", line)
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


# TODO: use utils.build_with_cmake
def build_spmv(args: argparse.Namespace, src: Path, specific_gen_args: List[str], target: str):
    clang = Path(environ['LLVM_PATH']) / "bin/clang"
    assert clang.exists()

    common_gen_args = [f"-DVARIANT={args.variant}", f"-DPREFETCH_DISTANCE={args.prefetch_distance}",
                       f"-DLOCALITY_HINT={args.locality_hint}", "-Bbuild", f"-S{src}"]

    gen_cmd = ["cmake", f"-DCMAKE_C_COMPILER={clang}"] + common_gen_args + specific_gen_args + ["-Bbuild", f"-S{src}"]
    run(gen_cmd, check=True)

    build_cmd = ["cmake", "--build", "build", "--target", target]
    run(build_cmd, check=True)


def build_lib(args: argparse.Namespace, src: Path, enable_logs: bool) -> Path:

    gen_args = []
    if enable_logs:
        gen_args.append("-DENABLE_LOGS=y")

    build_spmv(args, src, gen_args, "benchmark-spmv-runahead")

    lib_name = "libbenchmark-spmv-runahead" + (".dylib" if system() == "Darwin" else ".so")
    lib_path = Path("./build").resolve() / lib_name
    assert lib_path.exists(), f"Could not find {lib_name}"

    return lib_path


def build_exec(args: argparse.Namespace, src: Path) -> Path:

    build_spmv(args, src, [], "spmv-runahead")

    exec_name = "spmv-runahead"
    exec_path = Path("./build").resolve() / exec_name
    assert exec_path.exists(), f"Could not find {exec_name}"

    return exec_path


def parse_args() -> argparse.Namespace:
    common_arg_parser = get_spmv_arg_parser()
    common_arg_parser.add_argument("-v", "--variant", default="omp-tasks",
                                   help="Which variant of the spmv-runahead-* file to use")

    parser = argparse.ArgumentParser(description='(Sparse Matrix)x(Dense Vector) Multiplication (SpMV) '
                                                 'with runahead prefetching using OpenMP')

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")
    add_parser_for_profile(subparsers, parent_parser=common_arg_parser)
    add_parser_for_benchmark(subparsers, parent_parser=common_arg_parser)

    # check
    check_parser = subparsers.add_parser("check", parents=[common_arg_parser],
                                         help="Generate logs and perform various checks based on the logs")
    check_parser.add_argument("--enable-logs", action="store_true", help="Enable logs")

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


def check(args: argparse.Namespace, shared_lib: Path, mat: sp.csr_array, vec: np.ndarray):
    result, stdout, elapsed_wtime = run_spmv_as_foreign_fun(shared_lib, mat, vec)

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
    src_path = Path(__file__).parent.resolve() / "runahead-with-omp"
    assert src_path.exists()

    args = parse_args()

    mat, vec = create_sparse_mat_and_dense_vec(rows=args.rows, cols=args.cols, density=args.density, form="csr")

    make_work_dir_and_cd_to_it(__file__)

    if args.command == "check":
        lib = build_lib(args, src_path, enable_logs=True)
        check(args, lib, mat, vec)
    elif args.command == "benchmark":
        lib = build_lib(args, src_path, enable_logs=False)
        log_execution_times_secs(benchmark_spmv(args, lib, mat, vec))
    elif args.command == "profile":
        exe = build_exec(args, src_path)
        profile_spmv_with_vtune(exe, mat, vec, args.analysis)


if __name__ == "__main__":
    main()
