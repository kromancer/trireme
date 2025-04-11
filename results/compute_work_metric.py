import ast
import json
from pathlib import Path
from statistics import harmonic_mean
import sys
from typing import Dict, List, Tuple


def compute_work_metric(data: Dict[str, Dict], mtx_list: List[str]) -> Tuple[float, float]:
    all_llc_per_knnz = []
    all_nnz_per_ms = []
    for mtx in mtx_list:
        v = data[mtx]
        nnz = int(v["num_of_entries"])

        if "mem_load_uops_retired.dram_hit" in v:
            llc_per_knnz = (v["mem_load_uops_retired.dram_hit"] * 1000) / nnz
            v["llc_per_knnz"] = llc_per_knnz
            all_llc_per_knnz.append(llc_per_knnz)

        time_field = "mean_ms" if "mean_ms" in v else "time_ms"
        if time_field in v:
            nnz_per_ms = nnz / v[time_field]
            v["nnz_per_ms"] = nnz_per_ms
            all_nnz_per_ms.append(nnz_per_ms)

    ha_llc = harmonic_mean(all_llc_per_knnz) if all_llc_per_knnz else None
    ha_ms = harmonic_mean(all_nnz_per_ms) if all_nnz_per_ms else None
    return ha_llc, ha_ms


def compute_work_metrics_for_percentage(data: Dict[str, Dict], percentage: float) -> Tuple[float, float]:
    sorted_mtxs = sorted(
        data.items(),
        key=lambda kv: kv[1].get("num_of_entries", 0),
        reverse=True
    )
    keep_count = int(len(sorted_mtxs) * (percentage / 100.0))
    mtx_list = [k for k, _ in sorted_mtxs[:keep_count]]
    return compute_work_metric(data, mtx_list)


def compute_work_metrics_for_groups(data: Dict[str, Dict], groups: List[str]) -> Tuple[float, float]:
    mtx_list = [k for k, v in data.items() if v.get("group") in groups]
    return compute_work_metric(data, mtx_list)


if __name__ == "__main__":
    rep = Path(sys.argv[1])
    assert rep.exists(), f"{rep} does not exist"
    with open(rep, "r") as f, open(rep.parent / ("bak-" + rep.name), "w") as bak:
        dat = json.load(f)
        bak.write(json.dumps(dat, indent=4))

    arg2 = sys.argv[2] if len(sys.argv) > 2 else "100.0"
    try:
        p = float(arg2)
        assert 0 < p <= 100, "Percentage must be in (0, 100]"
        h_llc, h_ms = compute_work_metrics_for_percentage(dat, p)
    except ValueError:
        g = ast.literal_eval(arg2)
        h_llc, h_ms = compute_work_metrics_for_groups(dat, g)

    print(f"ha_llc: {h_llc}, ha_ms: {h_ms}")

    with open(rep, "w") as f:
        f.write(json.dumps(dat, indent=4))
