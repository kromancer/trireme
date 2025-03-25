import json

with open("no-opt.json", "r") as f, open("pref-mlir-45.json", "r") as g, open("pref-ains-45.json", "r") as h:
    no_opt = json.load(f)
    pref_mlir = json.load(g)
    pref_ains = json.load(h)

for mtx, data in no_opt.items():
    ins_count = data["inst_retired.any"]
    pref_mlir[mtx]["instr-overhead"] = pref_mlir[mtx]["inst_retired.any"] / ins_count
    pref_ains[mtx]["instr-overhead"] = pref_ains[mtx]["inst_retired.any"] / ins_count

with open("pref-mlir-45-2.json", "w") as f, open("pref-ains-45-2.json", "w") as g:
    f.write(json.dumps(pref_mlir, indent=4))
    g.write(json.dumps(pref_ains, indent=4))
