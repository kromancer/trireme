import json
from pathlib import Path
from statistics import harmonic_mean
import sys


from suite_sparse import SuiteSparse


def compute_hms_for_work_metrics(data):
    nnz_per_llc = []
    nnz_per_ms = []
    for mtx, data in data.items():
        nnz = int(ss.get_meta(mtx, "num_of_entries"))
        if "mem_load_uops_retired.dram_hit" in data:
            nnz_per_llc.append((data["mem_load_uops_retired.dram_hit"] * 1000) / nnz)

        time_field = "mean_ms" if "mean_ms" in data else "time_ms"
        if time_field in data:
            nnz_per_ms.append(nnz / data[time_field])

    ha_llc = None
    if len(nnz_per_llc) > 0:
        ha_llc = harmonic_mean(nnz_per_llc)

    ha_ms = None
    if len(nnz_per_ms) > 0:
        ha_ms = harmonic_mean(nnz_per_ms)

    return ha_llc, ha_ms


if __name__ == "__main__":
    rep = sys.argv[1]

    with open(rep, "r") as f:
        dat = json.load(f)

    ss = SuiteSparse(Path("."))

    base_llc, base_ms = compute_hms_for_work_metrics(dat)

    print("work metrics:", base_llc, base_ms)




