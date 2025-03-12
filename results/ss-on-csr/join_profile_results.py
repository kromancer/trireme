import re
import sys
import json

with open("profile-ss-csr-pref-mlir-45.json", "r") as p1, open("profile-2-ss-csr-pref-mlir-45.json", "r") as p2, open("consolidated-profile.json", "w") as consolidated:
    p1_dat = json.load(p1)
    p2_dat = json.load(p2)

    mtx_regex = re.compile(r"SuiteSparse (.*)$")
    for e1 in p1_dat:
        mtx1 = mtx_regex.search(e1["args"]).group(1)
        for e2 in p2_dat:
            mtx2 = mtx_regex.search(e2["args"]).group(1)
            if mtx1 == mtx2:
                e2["perf-stat"] += e1["perf-stat"]
                break

    consolidated.write(json.dumps(p2_dat, indent=4))
