from argparse import ArgumentParser
import json
import scipy.sparse.linalg as spla
from typing import Tuple

from tqdm import tqdm

from input_manager import InputManager
from suite_sparse import SuiteSparse


def normalize_bandwidth(mtx: str, lo: int, up: int, ss: SuiteSparse) -> Tuple[float, float]:
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


if __name__ == "__main__":
    parser = ArgumentParser(description="Compute scipy.sparse.linalg.spbandwidth on SuiteSparse matrices.")
    SuiteSparse.add_args(parser)
    args = parser.parse_args()

    ss = SuiteSparse(InputManager.get_working_dir("SuiteSparse"), args)

    args.matrix_format = "csr"
    args.in_source = "SuiteSparse"
    in_man = InputManager(args)

    bandwidth_dict = {}
    matrices = ss.get_matrices()
    with tqdm(total=len(matrices), desc=f"{args.kernel} on SuiteSparse") as pbar:
        for m_name in matrices:
            pbar.set_description(f"{args.kernel} on {m_name}")
            args.name = m_name
            sp_mat = in_man.get_ss_mat()

        try:
            lo, up = spla.spbandwidth(sp_mat)
            lo_norm, up_norm = normalize_bandwidth(m_name, lo, up, ss)
            bandwidth_dict[m_name] = {
                "low": lo,
                "low_norm": lo_norm,
                "upper": up,
                "upper_norm": up_norm
            }
        except Exception as e:
            print(f"Error processing {m_name}: {e}")

        pbar.update(1)

    with open("bandwidths.json", "w") as f:
        f.write(json.dumps(bandwidth_dict, indent=4))

