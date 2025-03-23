import json


with open("consolidated-pref-ains-45.json", "r") as f, open("consolidated-no-opt.json", "r") as g:
    pref = json.load(f)
    no_opt = json.load(g)

new_data = {}
for mtx, v in pref.items():
    new_data[mtx] = v
    new_data[mtx]["speed-up-no-opt"] = no_opt[mtx]["time_ms"] / v["time_ms"]

new_data = dict(sorted(new_data.items(), key=lambda e: e[1]["Info.Bottleneck.PCT_Load_Miss_Bound_Cycles"], reverse=True))
with open("cdf-speedups.json", "w") as f:
    f.write(json.dumps(new_data, indent=4))
