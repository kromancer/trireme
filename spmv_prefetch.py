import argparse
import ctypes

from jinja2 import Template
import numpy as np
from scipy.sparse import random as sparse_random

from mlir import ir
from mlir import runtime as rt
from mlir.dialects import sparse_tensor as st

from sparsifier import Sparsifier
from spmv import boilerplate


def render_template(rows: int, cols: int) -> str:
    with (open("./spmv.prefetch.mlir.jinja2", "r") as template_f,
          open("./spmv.prefetch.mlir", "w") as rendered_template_f):
        rendered_template = Template(template_f.read()).render(rows=rows, cols=cols)
        rendered_template_f.write(rendered_template)

    return rendered_template


def compile_and_run_spmv(module: ir.Module, compiler: Sparsifier, rows: int, cols: int):

    # Set up numpy input and buffer for output.
    density = 0.05
    a = sparse_random(rows, cols, density, dtype=np.float64).toarray()
    b = np.array(rows * [1.0], np.float64)
    c = np.zeros(cols, np.float64)

    # Compile and dump object file
    engine = compiler.compile(module)

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

    # dump jitted object file along with location of main function for debug purposes
    engine.dump_to_object_file("./dump.o")
    print(f"mlir_main addr: {hex(engine.raw_lookup('main'))}")
    print(f"spmv addr: { hex(engine.raw_lookup('spmv')) }")

    # Sanity check on computed result.
    expected = np.matmul(a, b)
    actual = rt.ranked_memref_to_numpy(mem_out[0])
    if np.allclose(actual, expected):
        pass
    else:
        quit(f"FAILURE")


def main(rows: int, cols: int):

    with ir.Context() as ctx, ir.Location.unknown():

        lvl_types = [st.DimLevelType.dense, st.DimLevelType.compressed]
        ordering = ir.AffineMap.get_permutation([0, 1])
        attr = st.EncodingAttr.get(lvl_types=lvl_types,
                                   dim_to_lvl=ordering,
                                   lvl_to_dim=ordering,
                                   pos_width=0,
                                   crd_width=0,
                                   context=ctx)

        compiler = Sparsifier(options="parallelization-strategy=none",
                              opt_level=0,
                              shared_libs=["/Users/ioanniss/llvm-project/build/lib/libmlir_c_runner_utils.dylib"])

        # Render the template, join it with the boilerplate and deserialize
        module = ir.Module.parse(render_template(rows, cols))
        spmv = str(module.operation.regions[0].blocks[0].operations[0].operation)
        module = ir.Module.parse(spmv + boilerplate(attr, rows, cols))

        compile_and_run_spmv(module=module, compiler=compiler, rows=rows, cols=cols)


def get_args():
    parser = argparse.ArgumentParser(description="Process rows and cols.")
    parser.add_argument("--rows", type=int, default=1024, help="Number of rows (default=1024)")
    parser.add_argument("--cols", type=int, default=1024, help="Number of columns (default=1024)")
    args = parser.parse_args()
    return args.rows, args.cols


if __name__ == "__main__":
    rows, cols = get_args()
    main(rows, cols)
