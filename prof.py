from argparse import Namespace
import json
from multiprocessing import shared_memory
from os import environ
from pathlib import Path
from platform import machine
from subprocess import run
from typing import Dict, List

from common import is_in_path, read_config
import numpy as np
import scipy.sparse as sp

from hwpref_controller import HwprefController
from log_plot import append_result


def compile_exe(spmv_ll: Path):
    clang = Path(environ['LLVM_PATH']) / "bin/clang"
    assert clang.exists()

    compile_spmv_cmd = [str(clang), "-O3", "-mavx2" if machine() == 'x86_64' else "",
                        "-Wno-override-module", "-c", str(spmv_ll), "-o", "spmv.o"]
    run(compile_spmv_cmd, check=True)

    main_fun = Path(__file__).parent.resolve() / "templates" / "spmv_csr.main.c"
    compile_exe_cmd = [str(clang), "-O0", "-fopenmp", "-g", str(main_fun), "spmv.o", "-o" "spmv"]
    run(compile_exe_cmd, check=True)


def gen_and_store_vtune_reports() -> None:
    reports = [
        {
            "args": ["hw-events", "-format=csv", "-csv-delimiter=comma", "-group-by=source-line"],
            "output": "vtune-hw-events.csv"
        },
        {
            "args": ["summary", "-format=text"],
            "output": "vtune-summary.txt"
        }
    ]

    db_entry = {}
    for report in reports:
        vtune_cmd = ["vtune", "-report"] + report["args"] + ["-report-output", report["output"]]
        run(vtune_cmd, check=True)

        with open(report["output"], "r") as f:
            db_entry[report["output"]] = f.read()

    append_result(db_entry)


def parse_perf_stat_json_output() -> List[Dict]:
    events = []  # To hold the successfully parsed dictionaries
    with open("perf-stat.json", "r") as f:
        for line in f:
            try:
                # Attempt to parse the line as JSON
                json_object = json.loads(line.strip())  # strip() to remove leading/trailing whitespace
                events.append(json_object)
            except json.JSONDecodeError:
                # If json.loads() raises an error, skip this line
                continue
    return events


def gen_and_store_perf_report() -> None:
    cmd = ["perf", "report", "--stdio"]
    report = run(cmd, check=True, text=True, capture_output=True)
    append_result({"perf-report": report.stdout})


def profile_spmv(args: Namespace, spmv_ll: Path, mat: sp.csr_array, vec: np.ndarray):
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

    post_run_action = None
    try:
        if args.analysis == "advisor":
            assert is_in_path("advisor")
            cmd = ["advisor"] + read_config("advisor-config.json", args.config) + ["--"] + spmv_cmd
        elif args.analysis == "vtune":
            assert is_in_path("vtune")
            cmd = ["vtune"] + read_config("vtune-config.json", args.config) + ["--"] + spmv_cmd
            post_run_action = gen_and_store_vtune_reports
        elif args.analysis == "perf":
            assert is_in_path("perf")
            cmd = ["perf"] + read_config("perf-config.json", args.config) + ["--"] + spmv_cmd
            post_run_action = gen_and_store_perf_report
        elif args.analysis == "toplev":
            assert is_in_path("toplev")
            cmd = ["toplev"] + read_config("toplev-config.json", args.config) + spmv_cmd
        else:
            print("Dry run")
            cmd = spmv_cmd

        with HwprefController(args):
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

    if post_run_action is not None:
        post_run_action()
