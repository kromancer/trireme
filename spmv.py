import argparse
import ctypes
import json
from os import getpid, environ, killpg
from pathlib import Path
import platform
import signal
import subprocess
from time import sleep
from typing import Dict, List, Optional, Tuple

import jinja2
import numpy as np
import scipy.sparse as sp

from mlir import runtime as rt
from mlir.execution_engine import *
from mlir import ir

from benchmark import add_parser_for_benchmark, benchmark
from common import Encodings, get_spmv_arg_parser, is_in_path, make_work_dir_and_cd_to_it, read_config
from generate_kernel import apply_passes
from hwpref_controller import HwprefController
from logging_and_graphing import append_result_to_db
from matrix_storage_manager import create_sparse_mat_and_dense_vec
from vtune import gen_and_store_reports


def render_templates(rows: int, cols: int, opt: Optional[str], pd: int, loc_hint: int) -> Tuple[str, str]:

    template_dir = Path(__file__).parent.resolve() / "templates"
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))

    template_names = {"pref-ains": "spmv.ainsworth.mlir.jinja2",
                      "pref-spe": "spmv.spe.mlir.jinja2",
                      "no-opt": "spmv.mlir.jinja2"}

    spmv_template = env.get_template(template_names[opt or "no-opt"])
    with open("spmv.mlir", "w") as f:
        if opt == "pref-ains" or opt == "pref-spe":
            spmv_rendered = spmv_template.render(rows=rows, cols=cols, pd=pd, loc_hint=loc_hint)
        else:
            spmv_rendered = spmv_template.render(rows=rows, cols=cols)
        f.write(spmv_rendered)

    # Function "main" will be injected after the sparse-assembler pass
    main_template = env.get_template("spmv.main.mlir.jinja2")
    main_rendered = main_template.render(rows=rows, cols=cols)

    return spmv_rendered, main_rendered


def run_spmv(llvm_mlir: ir.Module, args: argparse.Namespace, mat: sp.csr_array, vec: np.ndarray):
    llvm_path = environ.get("LLVM_PATH", None)
    if llvm_path is None:
        raise RuntimeError("Env var LLVM_PATH not specified")

    runtimes = ["libmlir_runner_utils", "libmlir_c_runner_utils"]
    shared_lib_suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    runtime_paths = [str(Path(llvm_path) / "lib" / (r + shared_lib_suffix)) for r in runtimes]

    B2_pos = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(mat.indptr)))
    B2_crd = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(mat.indices)))
    B_vals = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(mat.data)))

    c_vals = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(vec)))

    a = np.zeros(args.rows, np.float64)
    a_vals = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(a)))

    # Allocate a MemRefDescriptor to receive the output tensor.
    # The buffer itself is allocated inside the MLIR code generation.
    ref_out = rt.make_nd_memref_descriptor(1, ctypes.c_double)()
    mem_out = ctypes.pointer(ctypes.pointer(ref_out))

    exec_engine = ExecutionEngine(llvm_mlir, opt_level=3, shared_libs=runtime_paths)

    @benchmark(exec_engine, args)
    def run():
        nonlocal a, a_vals

        exec_engine.invoke("main", mem_out, B2_pos, B2_crd, B_vals, c_vals, a_vals)
        exec_engine.dump_to_object_file("spmm.o")

        # Sanity check on computed result.
        expected = mat.dot(vec)
        res = rt.ranked_memref_to_numpy(mem_out[0])
        assert np.allclose(res, expected), "Wrong output!"

        # reset output
        a = np.zeros(args.rows, np.float64)
        a_vals = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(a)))

    run()


def parse_perf_stat_json_output(report: str) -> List[Dict]:
    events = []  # To hold the successfully parsed dictionaries
    with open(report, "r") as f:
        for line in f:
            try:
                # Attempt to parse the line as JSON
                json_object = json.loads(line.strip())  # strip() to remove leading/trailing whitespace
                events.append(json_object)
            except json.JSONDecodeError:
                # If json.loads() raises an error, skip this line
                continue
    return events


# TODO: This is broken!
def profile(args: argparse.Namespace, llvm_mlir: ir.Module, mat: sp.csr_array, vec: np.ndarray):
    profiler: subprocess.Popen

    profile_cmd = []
    report = "report.txt"
    if args.analysis == "toplev":
        assert is_in_path("toplev")
        profile_cmd = ["toplev", "-l6", "--nodes", "/Backend_Bound.Memory_Bound*", "--user", "--json",
                       "-o", f"{report}", "--perf-summary", "perf.csv", "--pid"]
    elif args.analysis == "vtune":
        assert is_in_path("vtune")
        profile_cmd = ["vtune"] + read_config("vtune-config.json", "memory-access") + ["-target-pid"]
    elif args.analysis == "events":
        assert is_in_path("perf")
        events = read_config("perf-events.json", "events")
        profile_cmd = ["perf", "stat", "-e", ",".join(events), "-j", "-o", f"{report}", "--pid"]
    else:
        assert False, f"unknown analysis {args.analysis}"

    @ctypes.CFUNCTYPE(ctypes.c_void_p)
    def start_cb_for_profile():
        nonlocal profiler, profile_cmd

        spmv_pid = getpid()
        profiler = subprocess.Popen(profile_cmd + [f"{spmv_pid}"], start_new_session=True)

        # give ample of time to the profiling tool to boot
        sleep(15)

        print("kernel start")

    @ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint64)
    def stop_cb_for_profile(dur_ns: int):
        nonlocal profiler

        dur_ms = round(dur_ns / 1000000, 3)
        print(f"kernel finish: execution time: {dur_ms} ms")

        killpg(profiler.pid, signal.SIGINT)
        profiler.wait()

    run_spmv(llvm_mlir, args.rows, mat, vec, start_cb_for_profile, stop_cb_for_profile)

    if args.analysis == "toplev":
        with open(report, "r") as f:
            rep = json.loads(f.read())
    elif args.analysis == "events":
        rep = parse_perf_stat_json_output(report)
    else:
        rep = gen_and_store_reports()
    append_result_to_db({"report": rep})


def main():
    args = parse_args()
    make_work_dir_and_cd_to_it(__file__)
    spmv, main = render_templates(args.rows, args.cols, args.optimization, args.prefetch_distance, args.locality_hint)

    with ir.Context(), ir.Location.unknown():
        llvm_mlir, _ = apply_passes(spmv, kernel="spmv", pipeline="vect-vl4" if args.optimization == "vect-vl4" else "no-opt", main=main)

    mat, vec = create_sparse_mat_and_dense_vec(args.rows, args.cols, args.density, form=Encodings.CSR)

    with HwprefController(args):
        if args.command == "benchmark":
            run_spmv(llvm_mlir, args, mat, vec)
        elif args.command == "profile":
            profile(args, llvm_mlir, mat, vec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="(Sparse Matrix)x(Dense Vector) Multiplication (SpMV), "
                                                 "baseline and state-of-the-art sw prefetching, "
                                                 "from manually generated MLIR templates")

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # common args to all subcommands
    common_arg_parser = get_spmv_arg_parser()
    common_arg_parser.add_argument("-o", "--optimization",
                                   choices=["vect-vl4", "pref-ains", "pref-spe"],
                                   help="Use an optimized version of the kernel")
    HwprefController.add_args(common_arg_parser)

    add_parser_for_benchmark(subparsers, parent_parser=common_arg_parser)

    # TODO: re-use fun from common.py
    profile_parser = subparsers.add_parser("profile", parents=[common_arg_parser],
                                           help="Profile the application")
    profile_parser.add_argument("analysis", choices=["toplev", "vtune", "events"],
                                help="Choose an analysis type")

    return parser.parse_args()


if __name__ == "__main__":
    main()
