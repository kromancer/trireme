import json
from pathlib import Path
from statistics import harmonic_mean
import sys


from suite_sparse import SuiteSparse


if __name__ == "__main__":
    rep = Path(sys.argv[1])
    assert Path(rep).exists(), f"{rep} does not exist"

    with open(rep, "r") as f, open(rep.parent / ("bak-" + rep.name), "w") as bak:
        data = json.load(f)
        bak.write(json.dumps(data, indent=4))

    ss = SuiteSparse(Path("."))

    all_llc_per_knnz = []
    all_nnz_per_ms = []
    for mtx, v in data.items():
        nnz = int(ss.get_meta(mtx, "num_of_entries"))

        if "mem_load_uops_retired.dram_hit" in v:
            llc_per_knnz = (v["mem_load_uops_retired.dram_hit"] * 1000) / nnz
            v["llc_per_knnz"] = llc_per_knnz
            all_llc_per_knnz.append(llc_per_knnz)

        time_field = "mean_ms" if "mean_ms" in v else "time_ms"
        if time_field in v:
            nnz_per_ms = nnz / v[time_field]
            v["nnz_per_ms"] = nnz_per_ms
            all_nnz_per_ms.append(nnz_per_ms)

    ha_llc = None
    if len(all_llc_per_knnz) > 0:
        ha_llc = harmonic_mean(all_llc_per_knnz)

    ha_ms = None
    if len(all_nnz_per_ms) > 0:
        ha_ms = harmonic_mean(all_nnz_per_ms)

    print("work metrics:", ha_llc, ha_ms)

    with open(rep, "w") as f:
        f.write(json.dumps(data, indent=4))




