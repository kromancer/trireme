from argparse import Namespace
import json
import os
from pathlib import Path
from platform import machine
from subprocess import run
from typing import Dict, List

from common import is_in_path, read_config
from log_plot import append_result


def compile_exe(spmv_ll: Path):
    clang = Path(os.environ['LLVM_PATH']) / "bin/clang"
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


def profile_spmv(args: Namespace, spmv_ll: Path, nnz: int, buffers: List[str]):
    compile_exe(spmv_ll)

    spmv_cmd = ["./spmv", str(args.i), str(args.j), str(nnz)] + buffers

    post_run_action = None
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

    run(cmd, check=True)

    if post_run_action is not None:
        post_run_action()
