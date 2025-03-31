import json
import re

with open("pref-mlir-2MB-raw.json", "r") as f:
    data = json.load(f)

mtx_regex = re.compile(r"SuiteSparse (.*)$")

complete = []
for e in data:
    if e["status"] == "complete":
        mtx = mtx_regex.search(e["args"]).group(1)
        complete.append(mtx)

with open("complete.json", "w") as f:
    f.write(json.dumps(complete, indent=4))
