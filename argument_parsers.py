from argparse import ArgumentParser

import numpy as np

from common import SparseFormats


def add_dimension_args(parser: ArgumentParser, num_dims: int):
    dim_names = ["i", "j", "k", "l", "m", "n"]  # Extend this list if needed
    for dim in range(num_dims):
        dim_name = dim_names[dim]
        parser.add_argument(f"-{dim_name}", type=int, default=1024,
                            help=f"Size of dimension {dim_name} (default=1024)")


def add_sparse_format_arg(parser: ArgumentParser, tensor_name: str):
    parser.add_argument(f"--{tensor_name}-format", type=str, choices=[str(f) for f in SparseFormats],
                        default=str(SparseFormats.CSR),
                        help=f"Sparse storage format for {tensor_name}.")


def add_output_check_arg(parser: ArgumentParser):
    parser.add_argument("--check-output", action="store_true",
                        help="Check output by means of numpy/scipy. Warning: depending on input size, "
                             "it may crash by means of consuming all available memory")


def add_synth_tensor_arg(parser: ArgumentParser, num_dims=2):
    add_dimension_args(parser, num_dims)
    parser.add_argument("-d", "--density", type=float, default=0.05,
                        help="Density of the generated matrix: density equal to one means a full matrix, "
                             "density of 0 means a matrix with no non-zero items")
    add_dtype_arg(parser)


def add_dtype_arg(parser: ArgumentParser):
    # ensure that "types" can be converted to np.dtype
    types = [np.dtype("float64").name, np.dtype("float32"), np.dtype("int64").name,
             np.dtype("int32").name, np.dtype("bool")]
    parser.add_argument("--dtype", type=str, choices=[str(t) for t in types],
                        default=types[0],
                        help="Data type of the generated matrix")
    return parser


def add_opt_arg(parser: ArgumentParser):
    parser.add_argument("-o", "--optimization",
                        choices=["no-opt", "omp", "vect-vl4", "pref-mlir", "pref-mlir-omp",
                                 "pref-ains", "pref-ains-omp", "pref-spe"],
                        default="no-opt",
                        help="Use an optimized version of the kernel")


def get_spmm_arg_parser() -> ArgumentParser:
    parser = add_synth_tensor_arg(num_dims=3)
    parser.add_argument("--enable-prefetches", action='store_true', help='Enable prefetches')
    return parser


def add_args_for_benchmark(parser: ArgumentParser):
    parser.add_argument("--repetitions", type=int, default=5, help="Repeat the kernel with the same input")


def add_args_for_profile(parser: ArgumentParser):
    parser.add_argument("analysis", choices=["toplev", "vtune", "advisor", "perf", "dry-run"],
                        help="Choose an analysis type")
    parser.add_argument("config", type=str,
                        help="Tool's config, read from the corresponding .json")


def add_prefetch_distance_arg(parser: ArgumentParser):
    parser.add_argument("-pd", "--prefetch-distance", type=int, default=0)


def add_locality_hint_arg(parser: ArgumentParser):
    parser.add_argument("-l", "--locality-hint", type=int, choices=[0, 1, 2, 3], default=2,
                        help="Temporal locality hint for prefetch instructions, "
                             "3 for maximum temporal locality, 0 for no temporal locality. "
                             "On x86, value 3 will produce PREFETCHT0, while value 0 will produce PREFETCHNTA")
