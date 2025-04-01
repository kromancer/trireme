from argparse import Namespace
import json
from pathlib import Path
from subprocess import run
from typing import List

from common import is_in_path, read_config
from report_manager import ReportManager


def gen_and_store_vtune_reports(rep_man: ReportManager):
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

    rep_man.append_result(db_entry)


def parse_perf_stat_json_output(rep_man: ReportManager):
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
    rep_man.append_result({"perf-stat": events})


def gen_and_store_perf_record_report(rep_man: ReportManager):
    cmd = ["perf", "report", "--stdio"]
    report = run(cmd, check=True, text=True, capture_output=True)
    rep_man.append_result({"perf-record": report.stdout})


def profile_cmd(args: Namespace, cmd, rep_man: ReportManager):
    post_run_action = None
    if args.analysis == "advisor":
        assert is_in_path("advisor")
        cmd = ["advisor"] + read_config("advisor-config.json", args.config) + ["--"] + cmd
    elif args.analysis == "vtune":
        assert is_in_path("vtune")
        cmd = ["vtune"] + read_config("vtune-config.json", args.config) + ["--"] + cmd
        post_run_action = gen_and_store_vtune_reports
    elif args.analysis == "perf":
        assert is_in_path("perf")
        cmd = ["perf"] + read_config("perf-config.json", args.config) + ["--"] + cmd
        if args.config.startswith("record"):
            post_run_action = gen_and_store_perf_record_report
        else:
            assert args.config.startswith("stat")
            post_run_action = parse_perf_stat_json_output
    elif args.analysis == "toplev":
        assert is_in_path("toplev")
        cmd = ["toplev"] + read_config("toplev-config.json", args.config) + cmd
    else:
        print("Dry run")
        cmd = cmd

    run(cmd, check=True)

    if post_run_action is not None:
        post_run_action(rep_man)
