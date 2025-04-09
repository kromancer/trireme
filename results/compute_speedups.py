import argparse
import json
from pathlib import Path
from statistics import geometric_mean


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-b", "--baseline", type=str, required=True)
    argparser.add_argument("-o", "--optimized", type=str, required=True)
    args = argparser.parse_args()

    base_file = Path(args.baseline)
    assert base_file.exists(), f"{base_file} does not exist"
    optimized_file = Path(args.optimized)
    assert optimized_file.exists(), f"{optimized_file} does not exist"
    with open(base_file, "r") as b, open(optimized_file, "r") as o, open(optimized_file.parent / ("bak-" + optimized_file.name), "w") as bak:
        baseline = json.load(b)
        optimized = json.load(o)
        bak.write(json.dumps(optimized, indent=4))

    speedups = []
    for mtx, v in optimized.items():
        time_field = "time_ms" if "time_ms" in v else "mean_ms"
        v.setdefault("speed-ups", {})
        s = baseline[mtx][time_field] / v[time_field]
        v["speed-ups"][str(base_file.resolve())] = s
        speedups.append(s)

    new_data = dict(sorted(optimized.items(), key=lambda e: e[1]["speed-ups"][str(base_file.resolve())], reverse=True))
    with open(optimized_file, "w") as f:
        f.write(json.dumps(new_data, indent=4))

    print("geomean speed-up:", geometric_mean(speedups))
