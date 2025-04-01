import argparse
import json
from pathlib import Path


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-b", "--baseline", type=str, required=True)
    argparser.add_argument("-o", "--optimized", type=str, required=True)
    args = argparser.parse_args()

    opt = Path(args.optimized)
    assert opt.exists(), f"{opt} does not exist"
    with open(args.baseline, "r") as b, open(opt, "r") as o, open(opt.parent / ("bak-" + opt.name), "w") as bak:
        baseline = json.load(b)
        optimized = json.load(o)
        bak.write(json.dumps(optimized, indent=4))

    new_data = {}
    for mtx, v in optimized.items():
        new_data[mtx] = v
        time_field = "time_ms" if "time_ms" in v else "mean_ms"
        new_data[mtx]["speed-up"] = baseline[mtx][time_field] / v[time_field]

    new_data = dict(sorted(new_data.items(), key=lambda e: e[1]["speed-up"], reverse=True))
    with open(opt, "w") as f:
        f.write(json.dumps(new_data, indent=4))
