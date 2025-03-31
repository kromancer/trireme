import json
import os
from pathlib import Path
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys

from suite_sparse import SuiteSparse


def normalize_bandwidth(mtx, lo, up):
    ss = SuiteSparse(Path("."))
    rows = int(ss.get_meta(mtx, "num_of_rows"))
    cols = int(ss.get_meta(mtx, "num_of_cols"))
    low_norm = 0
    try:
        low_norm = lo / (rows - 1)
    except ZeroDivisionError:
        pass

    up_norm = 0
    try:
        up_norm = up / (cols - 1)
    except ZeroDivisionError:
        pass

    return low_norm, up_norm


def compute_bandwidths(path_to_ss, output):
    bandwidth_dict = {}

    for filename in os.listdir(path_to_ss):
        if filename.endswith(".npz"):
            file_path = os.path.join(path_to_ss, filename)
            mtx = os.path.splitext(filename)[0]  # Remove .npz extension

            try:
                matrix = sp.load_npz(file_path)
                lo, up = spla.spbandwidth(matrix)
                lo_norm, up_norm = normalize_bandwidth(mtx, lo, up)
                bandwidth_dict[mtx] = {
                    "low": lo,
                    "low_norm": lo_norm,
                    "upper": up,
                    "upper_norm": up_norm
                }
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    with open(output, "w") as f:
        f.write(json.dumps(bandwidth_dict, indent=4))


if __name__ == "__main__":
    directory = sys.argv[1]
    output_file = "bandwidths.json"
    compute_bandwidths(directory, output_file)
    print(f"Results saved in {output_file}")
