from argparse import ArgumentParser


def add_dimension_args(parser: ArgumentParser, num_dims: int):
    dim_names = ["i", "j", "k", "l", "m", "n"]  # Extend this list if needed
    for dim in range(num_dims):
        dim_name = dim_names[dim]
        parser.add_argument(f"-{dim_name}", type=int, default=1024,
                            help=f"Size of dimension {dim_name} (default=1024)")


def get_common_arg_parser(with_density=True, with_output_check=True, num_dims=2) -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    add_dimension_args(parser, num_dims)
    if with_density:
        parser.add_argument("-d", "--density", type=float, default=0.05,
                            help="Density of the generated matrix: density equal to one means a full matrix, "
                                 "density of 0 means a matrix with no non-zero items")

    if with_output_check:
        parser.add_argument("--enable-output-check", action="store_true",
                            help="Check output by means of numpy/scipy. Warning: depending on input size, "
                                 "it may crash by means of consuming all available memory")
    return parser


def get_spmv_arg_parser() -> ArgumentParser:
    parser = get_common_arg_parser(num_dims=2)
    parser.add_argument("-pd", "--prefetch-distance", type=int, default=32, help="Prefetch distance")
    parser.add_argument("-l", "--locality-hint", type=int, choices=[0, 1, 2, 3], default=0,
                        help="Temporal locality hint for prefetch instructions, "
                             "3 for maximum temporal locality, 0 for no temporal locality. "
                             "On x86, value 3 will produce PREFETCHT0, while value 0 will produce")
    return parser


def get_spmm_arg_parser() -> ArgumentParser:
    parser = get_common_arg_parser(num_dims=3)
    parser.add_argument("--enable-prefetches", action='store_true', help='Enable prefetches')
    return parser


def add_parser_for_benchmark(subparsers, parent_parser: ArgumentParser):
    benchmark_parser = subparsers.add_parser("benchmark", parents=[parent_parser],
                                             help="Benchmark the application.")
    benchmark_parser.add_argument("--repetitions", type=int, default=5,
                                  help="Repeat the kernel with the same input")


def add_parser_for_profile(subparsers, parent_parser):
    profile_parser = subparsers.add_parser("profile", parents=[parent_parser],
                                           help="Profile the application using vtune")
    profile_parser.add_argument("vtune_cfg", type=str, help="Choose an analysis type")
