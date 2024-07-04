from argparse import ArgumentParser, Namespace
import ctypes

from mlir.execution_engine import ExecutionEngine

from logging_and_graphing import log_execution_times_ns


def add_parser_for_benchmark(subparsers, parent_parser: ArgumentParser):
    benchmark_parser = subparsers.add_parser("benchmark", parents=[parent_parser],
                                             help="Benchmark the application.")
    benchmark_parser.add_argument("--repetitions", type=int, default=5,
                                  help="Repeat the kernel with the same input. Gather execution times stats")


def benchmark(exec_engine: ExecutionEngine, args: Namespace):
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

    exec_engine.register_runtime("start_measurement_callback", start_cb)
    exec_engine.register_runtime("stop_measurement_callback", stop_cb)

    def decorator(func):
        def wrapper(*func_args, **func_kwargs):
            nonlocal execution_times
            execution_times = []
            for _ in range(args.repetitions):
                func(*func_args, **func_kwargs)
            log_execution_times_ns(execution_times)

        return wrapper

    return decorator
