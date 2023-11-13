import ctypes
import time

import numpy as np
from scipy.sparse import random as sparse_random

from mlir import ir
from mlir import runtime as rt
from mlir.dialects import func
from mlir.dialects import sparse_tensor as st
from mlir.dialects.linalg.opdsl import lang as dsl

import sparsifier


def boilerplate(attr: st.EncodingAttr, rows: int, cols: int):
    """Returns boilerplate main method.

    This method sets up a boilerplate main method that takes three tensors
    (a, b, c), converts the first tensor a into s sparse tensor, and then
    calls the sparse kernel for matrix multiplication. For convenience,
    this part is purely done as string input.
    """
    return f"""
func.func @main(%ad: tensor<{rows}x{cols}xf64>, %b: tensor<{rows}xf64>, %c: tensor<{cols}xf64>) -> tensor<{cols}xf64>
  attributes {{ llvm.emit_c_interface }} {{
  %a = sparse_tensor.convert %ad : tensor<{rows}x{cols}xf64> to tensor<{rows}x{cols}xf64, {attr}>
  %0 = call @spMV(%a, %b, %c) : (tensor<{rows}x{cols}xf64, {attr}>,
                                  tensor<{rows}xf64>,
                                  tensor<{cols}xf64>) -> tensor<{cols}xf64>
  return %0 : tensor<{cols}xf64>
}}
"""


@dsl.linalg_structured_op
def matvec_dsl(
        A=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.K),
        B=dsl.TensorDef(dsl.T, dsl.S.K),
        C=dsl.TensorDef(dsl.T, dsl.S.M, output=True),
):
    C[dsl.D.m] += A[dsl.D.m, dsl.D.k] * B[dsl.D.k]


def build_SpMV(attr: st.EncodingAttr, rows: int, cols: int):
    module = ir.Module.create()
    f64 = ir.F64Type.get()
    a = ir.RankedTensorType.get([rows, cols], f64, attr)
    b = ir.RankedTensorType.get([rows], f64)
    c = ir.RankedTensorType.get([cols], f64)
    arguments = [a, b, c]

    with ir.InsertionPoint(module.body):
        @func.FuncOp.from_py_func(*arguments)
        def spMV(*args):
            return matvec_dsl(args[0], args[1], outs=[args[2]])

    return module


def build_compile_and_run_SpMV(attr: st.EncodingAttr, compiler):

    # Set up numpy input and buffer for output.
    rows, cols = 1024, 1024
    density = 0.05
    a = sparse_random(rows, cols, density, dtype=np.float64).toarray()
    b = np.array(rows * [1.0], np.float64)
    c = np.zeros(cols, np.float64)

    # Build.
    module = build_SpMV(attr, rows, cols)
    func = str(module.operation.regions[0].blocks[0].operations[0].operation)
    module = ir.Module.parse(func + boilerplate(attr, rows, cols))

    # Compile.
    engine = compiler.compile_and_jit(module)

    mem_a = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(a)))
    mem_b = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(b)))
    mem_c = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(c)))

    # Allocate a MemRefDescriptor to receive the output tensor.
    # The buffer itself is allocated inside the MLIR code generation.
    ref_out = rt.make_nd_memref_descriptor(1, ctypes.c_double)()
    mem_out = ctypes.pointer(ctypes.pointer(ref_out))

    # Invoke the kernel and get numpy output.
    # Built-in bufferization uses in-out buffers.
    engine.invoke("main", mem_out, mem_a, mem_b, mem_c)

    # Sanity check on computed result.
    expected = np.matmul(a, b)
    actual = rt.ranked_memref_to_numpy(mem_out[0])
    if np.allclose(actual, expected):
        pass
    else:
        quit(f"FAILURE")


def main():

    with ir.Context() as ctx, ir.Location.unknown():

        lvl_types = [st.DimLevelType.dense, st.DimLevelType.compressed]
        ordering = ir.AffineMap.get_permutation([0, 1])
        attr = st.EncodingAttr.get(lvl_types=lvl_types,
                                   dim_to_lvl=ordering,
                                   lvl_to_dim=ordering,
                                   pos_width=0,
                                   crd_width=0,
                                   context=ctx)

        compiler = sparsifier.Sparsifier(
            options="parallelization-strategy=none",
            opt_level=0,
            shared_libs=["/Users/ioanniss/llvm-project/build/lib/libmlir_c_runner_utils.dylib"]
        )

        build_compile_and_run_SpMV(attr, compiler)


if __name__ == "__main__":
    main()
