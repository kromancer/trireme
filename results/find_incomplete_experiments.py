import json
import re
import sys


if __name__ == "__main__":
    rep = sys.argv[1]
    with open(rep, "r") as f:
        data = json.load(f)

    mtx_regex = re.compile(r"SuiteSparse (.*)$")

    complete = []
    for e in data:
        if e["status"] != "complete":
            mtx = mtx_regex.search(e["args"]).group(1)
            complete.append(mtx)

    with open("incomplete.json", "w") as f:
        f.write(json.dumps(complete, indent=4))
