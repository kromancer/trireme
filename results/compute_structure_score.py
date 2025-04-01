import argparse
import numpy as np

from scipy.sparse import csr_array
from scipy.stats import entropy

from tqdm import tqdm

from input_manager import InputManager
from suite_sparse import SuiteSparse
from report_manager import create_report_manager, ReportManager


def sparse_structure_score(matrix: csr_array, is_symmetric: bool):
    if is_symmetric:
        # Reconstruct full symmetric matrix from lower triangle
        # Since we only care about the shape, avoid subtracting the diag
        matrix += matrix.transpose()

    n_rows, n_cols = matrix.shape

    # 1. Row histogram
    row_nnz = np.diff(matrix.indptr)
    row_probs = row_nnz / row_nnz.sum()
    row_entropy = entropy(row_probs, base=np.e) / np.log(n_rows)

    # 2. Column histogram
    col_nnz = np.bincount(matrix.indices, minlength=n_cols)
    col_probs = col_nnz / col_nnz.sum()
    col_entropy = entropy(col_probs, base=np.e) / np.log(n_cols)

    # 3. Diagonal histogram
    max_offset = n_rows - 1
    min_offset = -(n_cols - 1)
    offsets = np.arange(min_offset, max_offset + 1)
    diag_nnz = np.zeros_like(offsets, dtype=int)

    coo = matrix.tocoo()
    diag_offsets = coo.row - coo.col
    offset_to_index = {offset: i for i, offset in enumerate(offsets)}
    for d in diag_offsets:
        diag_nnz[offset_to_index[d]] += 1

    diag_probs = diag_nnz / diag_nnz.sum()
    diag_entropy = entropy(diag_probs, base=np.e) / np.log(len(offsets))

    # Compute sparsity
    density = matrix.nnz / (n_rows * n_cols)

    # Final structure score
    structure_score = 1 - min(row_entropy, col_entropy, diag_entropy)
    final_score = density + (1-density) * structure_score

    return {
        "density": density,
        "structure_score": structure_score,
        "final_score": final_score,
        "row_entropy": row_entropy,
        "col_entropy": col_entropy,
        "diag_entropy": diag_entropy
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute structure score for SuiteSparse matrices",)
    SuiteSparse.add_args(parser)
    ReportManager.add_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    args.in_source = "SuiteSparse"
    args.matrix_format = "csr"
    in_man = InputManager(args)
    rep_man = create_report_manager(args)
    ss = SuiteSparse(InputManager.get_working_dir("SuiteSparse"), args)

    matrix_names = ss.get_matrices()
    with tqdm(total=len(matrix_names), desc="structure score on SuiteSparse") as pbar:
        for mtx in matrix_names:
            rep_man.append_placeholder(mtx)
            args.name = mtx
            m: csr_array = in_man.get_ss_mat()
            structure_metrics = sparse_structure_score(m, args.symmetric)
            rep_man.append_result(structure_metrics)
            pbar.update(1)


if __name__ == "__main__":
    main()
