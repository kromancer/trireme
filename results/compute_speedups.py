import argparse
from statistics import geometric_mean

from common import json_load, json_load_and_backup, json_store

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-b", "--baseline", type=str, required=True)
    argparser.add_argument("-o", "--optimized", type=str, required=True)
    args = argparser.parse_args()

    baseline = json_load(args.baseline)
    optimized = json_load_and_backup(args.optimized)

    speedups = []
    for mtx, v in optimized.items():
        time_field = "time_ms" if "time_ms" in v else "mean_ms"
        s = baseline[mtx][time_field] / v[time_field]
        v["speedup"] = s
        speedups.append(s)

    json_store(args.optimized, optimized)

    print("geomean speed-up:", geometric_mean(speedups))
