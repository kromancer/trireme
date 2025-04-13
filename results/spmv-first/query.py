from pathlib import Path
import json

from suite_sparse import SuiteSparse

with open("consolidated-no-opt.json", "r") as no_opt, open("consolidated-pref-mlir-45.json", "r") as pref_mlir, open("consolidated-pref-ains-45.json", "r") as pref_ains:
    json_no_opt = json.load(no_opt)
    json_pref_mlir = json.load(pref_mlir)
    json_pref_ains = json.load(pref_ains)

total = 0
better = 0
close = 0
ss = SuiteSparse(Path("."))
candidates = []
for mtx, data in json_pref_mlir.items():
    total += 1
    if data["time_ms"] < json_no_opt[mtx]["time_ms"] and data["time_ms"] < json_pref_ains[mtx]["time_ms"]:
        better += 1
        if ss.get_meta(mtx, "is_binary") != 1 and ss.get_meta(mtx, "psym") != 1:
            candidates.append((mtx, json_no_opt[mtx]["time_ms"]/data["time_ms"]))

print(sorted(candidates, key=lambda e: e[1], reverse=True))
print("total: ", total)
print("better: ", better)
print("close: ", close)
