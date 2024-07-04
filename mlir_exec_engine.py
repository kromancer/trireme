import platform
from os import environ
from pathlib import Path

from mlir import ir
import mlir.execution_engine as mlir_execution_engine


def create_exec_engine(module: ir.Module) -> mlir_execution_engine.ExecutionEngine:
    llvm_path = environ.get("LLVM_PATH", None)
    if llvm_path is None:
        raise RuntimeError("Env var LLVM_PATH not specified")

    runtimes = ["libmlir_runner_utils", "libmlir_c_runner_utils"]
    shared_lib_suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    runtime_paths = [str(Path(llvm_path) / "lib" / (r + shared_lib_suffix)) for r in runtimes]
    return mlir_execution_engine.ExecutionEngine(module, opt_level=3, shared_libs=runtime_paths)
