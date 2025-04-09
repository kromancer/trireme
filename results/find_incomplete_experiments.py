import json
from pathlib import Path
import re
import sys


if __name__ == "__main__":
    rep = Path(sys.argv[1])
    assert Path(rep).exists(), f"{rep} does not exist"

    with open(rep, "r") as f, open(rep.parent / ("bak-" + rep.name), "w") as bak:
        data = json.load(f)
        bak.write(json.dumps(data, indent=4))

    mtx_regex = re.compile(r"SuiteSparse (.*)$")

    complete_mtx_names = []
    incomplete_mtx_names = []
    complete = []
    for e in data:
        mtx = mtx_regex.search(e["args"]).group(1)
        if e["status"] != "complete":
            incomplete_mtx_names.append(mtx)
        else:
            complete_mtx_names.append(mtx)
            complete.append(e)

    with (open(rep.parent / ("incomplete-" + rep.name), "w") as inc,
          open(rep.parent / ("complete-" + rep.name), "w") as comp, open(rep, "w") as f):
        inc.write(json.dumps(incomplete_mtx_names, indent=4))
        comp.write(json.dumps(complete_mtx_names, indent=4))
        f.write(json.dumps(complete, indent=4))
