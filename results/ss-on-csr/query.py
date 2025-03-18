from pathlib import Path
import json

from suite_sparse import SuiteSparse

with open("consolidated-no-opt.json", "r") as no_opt, open("consolidated-pref-mlir-45.json", "r") as pref_mlir:
    json_no_opt = json.load(no_opt)
    json_pref_mlir = json.load(pref_mlir)

total = 0
better = 0
close = 0
ss = SuiteSparse(Path("."))
candidates = []
for mtx, data in json_no_opt.items():
    total += 1
    if data["time_ms"] > json_pref_mlir[mtx]["time_ms"]:
        better += 1
        if ss.get_meta(mtx, "is_binary") != 1:
            candidates.append((mtx, data["time_ms"]/json_pref_mlir[mtx]["time_ms"]))
    elif data["time_ms"]/json_pref_mlir[mtx]["time_ms"] >= 0.90:
        # print(mtx, data["cols"])
        close += 1

print(sorted(candidates, key=lambda e: e[1], reverse=True))
print("total: ", total)
print("better: ", better)
print("close: ", close)
