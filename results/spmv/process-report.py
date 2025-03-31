import json
import re
import sys


if __name__ == "__main__":
    rep = sys.argv[1]
    with open(rep, "r") as f:
        data = json.load(f)

    mtx_regex = re.compile(r"SuiteSparse (.*)$")

    proced = {}
    for e in data:
        mtx = mtx_regex.search(e["args"]).group(1)
        proced[mtx] = e

    with open(f"{rep}.proced.json", "w") as f:
        f.write(json.dumps(proced, indent=4))

