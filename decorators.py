from argparse import Namespace
import ctypes
import json
from os import getpid, killpg
import signal
from subprocess import Popen
from time import sleep
from typing import Callable, Dict, List, TypeVar

from mlir.execution_engine import ExecutionEngine

from common import is_in_path, read_config
from logging_and_graphing import append_result_to_db, log_execution_times_ns
from vtune import gen_and_store_reports

RunFuncType = TypeVar("RunFuncType", bound=Callable[...,  None])


def benchmark(exec_engine: ExecutionEngine, args: Namespace) -> Callable[[RunFuncType], RunFuncType]:
    assert hasattr(args, 'repetitions'), "The args namespace must contain 'repetitions'."
    execution_times = []

    @ctypes.CFUNCTYPE(ctypes.c_void_p)
    def start_cb():
        print("kernel start")

    @ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint64)
    def stop_cb(dur_ns: int):
        nonlocal execution_times

        dur_ms = round(dur_ns / 1000000, 3)
        print(f"kernel finish: execution time: {dur_ms} ms")
        execution_times.append(dur_ns)

    exec_engine.register_runtime("start_cb", start_cb)
    exec_engine.register_runtime("stop_cb", stop_cb)

    def decorator(func: RunFuncType) -> RunFuncType:
        def wrapper(*func_args, **func_kwargs):
            nonlocal execution_times
            execution_times = []
            for _ in range(args.repetitions):
                func(*func_args, **func_kwargs)
            log_execution_times_ns(execution_times)

        return wrapper

    return decorator


def get_profile_cmd(args: Namespace, report: str) -> List[str]:
    if args.analysis == "toplev":
        assert is_in_path("toplev")
        return ["toplev", "-l6", "--nodes", "/Backend_Bound.Memory_Bound*", "--user", "--json", "-o", f"{report}",
                "--perf-summary", "perf.csv", "--pid"]
    elif args.analysis == "vtune":
        assert is_in_path("vtune")
        return ["vtune"] + read_config("vtune-config.json", "memory-access") + ["-target-pid"]
    elif args.analysis == "events":
        assert is_in_path("perf")
        events = read_config("perf-events.json", "events")
        return ["perf", "stat", "-e", ",".join(events), "-j", "-o", f"{report}", "--pid"]
    else:
        assert False, f"unknown analysis {args.analysis}"


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


def profile(exec_engine: ExecutionEngine, args: Namespace) -> Callable[[RunFuncType], RunFuncType]:
    assert hasattr(args, 'analysis'), "The args namespace must contain 'analysis'."

    profiler: Popen
    profile_cmd = []
    report = "report.txt"
    profile_cmd = get_profile_cmd(args, report)

    @ctypes.CFUNCTYPE(ctypes.c_void_p)
    def start_cb():
        nonlocal profiler, profile_cmd

        spmv_pid = getpid()
        profiler = Popen(profile_cmd + [f"{spmv_pid}"], start_new_session=True)

        # give ample of time to the profiling tool to boot
        sleep(15)

        print("kernel start")

    @ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint64)
    def stop_cb(dur_ns: int):
        nonlocal profiler

        dur_ms = round(dur_ns / 1000000, 3)
        print(f"kernel finish: execution time: {dur_ms} ms")

        killpg(profiler.pid, signal.SIGINT)
        profiler.wait()

    exec_engine.register_runtime("start_cb", start_cb)
    exec_engine.register_runtime("stop_cb", stop_cb)

    def decorator(func: RunFuncType) -> RunFuncType:
        def wrapper(*func_args, **func_kwargs):
            func(*func_args, **func_kwargs)
            if args.analysis == "toplev":
                with open(report, "r") as f:
                    rep = json.loads(f.read())
            elif args.analysis == "events":
                rep = parse_perf_stat_json_output(report)
            else:
                rep = gen_and_store_reports()
            append_result_to_db({"report": rep})

        return wrapper

    return decorator
