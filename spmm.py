import argparse

from common import add_parser_for_benchmark, Encodings, get_spmm_arg_parser, make_work_dir_and_cd_to_it
from matrix_storage_manager import create_sparse_mat
from generate_kernel import apply_passes, make_spmm_mlir_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="(Sparse Matrix)x(Sparse Matrix) Multiplication (SpMM)")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    common_arg_parser = get_spmm_arg_parser()
    add_parser_for_benchmark(subparsers, parent_parser=common_arg_parser)

    return parser.parse_args()


def main():
    args = parse_args()
    make_work_dir_and_cd_to_it(__file__)

    enc_B = Encodings.CSR
    B = create_sparse_mat(args.rows, args.inner_dim_size, args.density, enc_B)

    enc_C = Encodings.CSR
    C = create_sparse_mat(args.inner_dim_size, args.cols, args.density, enc_C)

    src = str(make_spmm_mlir_module(args.rows, args.cols, args.inner_dim_size, enc_B, enc_C))
    llvm_mlir = apply_passes(kernel="spmm", src=src, pipeline="no-opt")


if __name__ == "__main__":
    main()




