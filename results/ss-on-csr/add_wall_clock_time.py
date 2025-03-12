import re
import sys
import json

with open(sys.argv[1], "r") as bench, open(sys.argv[2], "r") as prof, open("consolidated.json", "w") as consolidated:
    bench_dat = json.load(bench)
    prof_dat = json.load(prof)

    mtx_regex = re.compile(r"SuiteSparse (.*)$")
    for e in bench_dat:
        mtx = mtx_regex.search(e["args"]).group(1)
        prof_dat[mtx]["time_ms"] = e["mean_ms"]
        # assert e["cv"] < 0.2

    consolidated.write(json.dumps(prof_dat, indent=4))
