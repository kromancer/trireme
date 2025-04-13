import json
import re


with open("no-opt.json", "r") as f, open("pref-mlir-45.json", "r") as g, open("pref-ains-45.json", "r") as h:
    no_opt_before = json.load(f)
    mlir_before = json.load(g)
    ains_before = json.load(h)

with open("no-opt-br.json", "r") as f, open("pref-mlir-br.json", "r") as g, open("pref-ains-br.json", "r") as h:
    no_opt_unparsed = json.load(f)
    mlir_unparsed = json.load(g)
    ains_unparsed = json.load(h)

mtx_regex = re.compile(r"SuiteSparse (.*)$")


def parse_new_prof(data):
    new_data = {}
    for d in data:
        mtx = mtx_regex.search(d["args"]).group(1)
        new_data[mtx] = {}
        for e in d["perf-stat"]:
            if e["event"].startswith("cpu_atom/"):
                event_name = e["event"].replace("cpu_atom/", "").replace(":u/", "")
                new_data[mtx][event_name] = float(e["counter-value"])
    return new_data


no_opt_new = parse_new_prof(no_opt_unparsed)
mlir_new = parse_new_prof(mlir_unparsed)
ains_new = parse_new_prof(ains_unparsed)


def add_to_before(before, new):
    for mtx, events in new.items():
        before[mtx] |= events


add_to_before(no_opt_before, no_opt_new)
add_to_before(mlir_before, mlir_new)
add_to_before(ains_before, ains_new)

with open("no-opt.json", "w") as f, open("pref-mlir-45.json", "w") as g, open("pref-ains-45.json", "w") as h:
    f.write(json.dumps(no_opt_before, indent=4))
    g.write(json.dumps(mlir_before, indent=4))
    h.write(json.dumps(ains_before, indent=4))
