import argparse
import json


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-b", "--baseline", type=str, required=True)
    argparser.add_argument("-o", "--optimized", type=str, required=True)
    args = argparser.parse_args()

    with open(args.baseline, "r") as b, open(args.optimized, "r") as o:
        baseline = json.load(b)
        optimized = json.load(o)

    new_data = {}
    for mtx, v in optimized.items():
        new_data[mtx] = v
        time_field = "time_ms" if "time_ms" in v else "mean_ms"
        new_data[mtx]["speed-up"] = baseline[mtx][time_field] / v[time_field]

    new_data = dict(sorted(new_data.items(), key=lambda e: e[1]["speed-up"], reverse=True))
    with open(f"{args.optimized}.speedups.json", "w") as f:
        f.write(json.dumps(new_data, indent=4))
