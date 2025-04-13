from collections import defaultdict
from common import json_load
from scipy.stats import hmean
import sys
from typing import Dict


def groups_with_all_speedup_ge_1(data: Dict[str, Dict]):
    groups = defaultdict(list)
    for name, vals in data.items():
        groups[vals["group"]].append(vals["speedup-mlir-pref"])

    return [g for g, speeds in groups.items() if all(s >= 1 for s in speeds)]


def groups_with_agg_speedup_ge_1(data: Dict[str, Dict]):
    nnz_base = defaultdict(list)
    nnz_mlir = defaultdict(list)

    for _, vals in data.items():
        group = vals["group"]
        nnz_base[group].append(vals["nnz_per_ms"])
        nnz_mlir[group].append(vals["nnz_per_ms_mlir-pref"])

    results = []
    for group in nnz_base:
        h_base = hmean(nnz_base[group])
        h_mlir = hmean(nnz_mlir[group])
        agg_speedup = h_mlir / h_base
        if agg_speedup >= 1:
            results.append((group, agg_speedup))

    for group, speedup in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{group}: {speedup:.3f}")


if __name__ == "__main__":
    data = json_load(sys.argv[1])
    # print(groups_with_all_speedup_ge_1(data))
    groups_with_agg_speedup_ge_1(data)
