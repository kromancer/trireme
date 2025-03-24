from pathlib import Path
from statistics import harmonic_mean
import json

from suite_sparse import SuiteSparse

with open("no-opt.json", "r") as f, open("pref-mlir-45.json", "r") as m, open("pref-ains-45.json", "r") as a:
    no_opt = json.load(f)
    pref_mlir = json.load(m)
    pref_ains = json.load(a)

assert len(no_opt) == len(pref_mlir) == len(pref_ains)

ss = SuiteSparse(Path("."))


def compute_hms_for_work_metrics(prof_data):
    nnz_per_llc = []
    nnz_per_ms = []
    for mtx, data in prof_data.items():
        nnz = int(ss.get_meta(mtx, "num_of_entries"))
        nnz_per_llc.append((data["mem_load_uops_retired.dram_hit"] * 1000) / nnz)
        nnz_per_ms.append(nnz / data["time_ms"])

    return harmonic_mean(nnz_per_llc), harmonic_mean(nnz_per_ms)


base_llc, base_ms = compute_hms_for_work_metrics(no_opt)
mlir_llc, mlir_ms = compute_hms_for_work_metrics(pref_mlir)
ains_llc, ains_ms = compute_hms_for_work_metrics(pref_ains)

print("Baseline:", base_llc, base_ms)
print("MLIR-Pref:", mlir_llc, mlir_ms)
print("Pref-Ains:", ains_llc, ains_ms)




