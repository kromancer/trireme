from argparse import Namespace
import ctypes
from typing import Callable, TypeVar

from mlir.execution_engine import ExecutionEngine

from log_plot import log_execution_times_ns

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
