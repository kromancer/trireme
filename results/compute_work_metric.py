import json
from pathlib import Path
from statistics import harmonic_mean
import sys


if __name__ == "__main__":
    rep = Path(sys.argv[1])
    percentage = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0

    assert rep.exists(), f"{rep} does not exist"
    assert 0 < percentage <= 100, "Percentage must be in (0, 100]"

    with open(rep, "r") as f, open(rep.parent / ("bak-" + rep.name), "w") as bak:
        data = json.load(f)
        bak.write(json.dumps(data, indent=4))

    # sort by nnz, keep top percentage
    sorted_mtxs = sorted(
        data.items(),
        key=lambda kv: kv[1].get("num_of_entries", 0),
        reverse=True
    )
    keep_count = int(len(sorted_mtxs) * (percentage / 100.0))
    top_mtxs = [k for k, _ in sorted_mtxs[:keep_count]]

    all_llc_per_knnz = []
    all_nnz_per_ms = []
    for mtx in top_mtxs:
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

    print(f"work metrics for top {percentage:.1f}% matrices:", ha_llc, ha_ms)

    with open(rep, "w") as f:
        f.write(json.dumps(data, indent=4))
