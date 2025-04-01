import json
from pathlib import Path
import re
import sys


if __name__ == "__main__":
    rep = Path(sys.argv[1])
    assert rep.exists(), f"{rep} does not exist"
    with open(rep, "r") as f, open(rep.parent / ("bak-" + rep.name), "w") as b:
        data = json.load(f)
        b.write(json.dumps(data, indent=4))

    mtx_regex = re.compile(r"SuiteSparse (.*)$")

    proced = {}
    for e in data:
        mtx = mtx_regex.search(e["args"]).group(1)
        proced[mtx] = e

    with open(rep, "w") as f:
        f.write(json.dumps(proced, indent=4))

