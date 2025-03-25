import json


with open("no-opt.json", "r") as f, open("pref-mlir-45.json", "r") as g, open("pref-ains-45.json", "r") as h:
    no_opt = json.load(f)
    pref_mlir = json.load(g)
    pref_ains = json.load(h)


def compute_ipbranch(data):
    for d in data.values():
        d["Info.Br_Inst_Mix.IpBranch"] = d["inst_retired.any"] / d["br_inst_retired.all_branches"]


for v in [no_opt, pref_mlir, pref_ains]:
    compute_ipbranch(v)

with open("no-opt.json", "w") as f, open("pref-mlir-45.json", "w") as g, open("pref-ains-45.json", "w") as h:
    f.write(json.dumps(no_opt, indent=4))
    g.write(json.dumps(pref_mlir, indent=4))
    h.write(json.dumps(pref_ains, indent=4))
