from pathlib import Path
from statistics import harmonic_mean
import json

from suite_sparse import SuiteSparse

with open("consolidated-no-opt.json", "r") as f, open("consolidated-pref-mlir-45.json", "r") as g:
    no_opt = json.load(f)
    pref_mlir = json.load(g)

ss = SuiteSparse(Path("."))
work_metrics_no_opt = []
for mtx, data in no_opt.items():
    nnz = int(ss.get_meta(mtx, "num_of_entries"))
    work_metrics_no_opt.append(nnz / data["time_ms"])

work_metrics_pref_mlir = []
for mtx, data in pref_mlir.items():
    nnz = int(ss.get_meta(mtx, "num_of_entries"))
    work_metrics_pref_mlir.append(nnz / data["time_ms"])

print("HA no_opt:", harmonic_mean(work_metrics_no_opt))
print("HA pref_mlir:", harmonic_mean(work_metrics_pref_mlir))




