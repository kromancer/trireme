import json
from pathlib import Path
import sys


if __name__ == "__main__":
    bench = Path(sys.argv[1])
    prof = Path(sys.argv[2])
    assert bench.exists(), f"{bench} does not exist"
    assert prof.exists(), f"{prof} does not exist"

    with open(bench, "r") as b, open(prof, "r") as p, open(bench.parent / ("bak-" + bench.name), "w") as bak:
        bench_data = json.load(b)
        prof_data = json.load(p)
        bak.write(json.dumps(bench_data, indent=4))

    for mtx, v in bench_data.items():
        for e in prof_data[mtx]["perf-stat"]:
            try:
                count = float(e["counter-value"])
            except ValueError:
                continue
            v[e["event"].replace("cpu_atom/", "").replace(":u/", "")] = count

    with open(bench, "w") as f:
        f.write(json.dumps(bench_data, indent=4))

