import json

with open("consolidated-no-opt.json", "r") as f, open("consolidated-pref-mlir-45.json", "r") as g, open("consolidated-pref-ains-45.json", "r") as h:
    no_opt_bef = json.load(f)
    mlir_bef = json.load(g)
    ains_bef = json.load(h)


def compute_extra_metrics(data):
    for mtx, events in data.items():
        clks = events["cpu_clk_unhalted.core"]
        slots = 5 * clks
        events["Topdown.Backend_Bound.Resource_Bound.Serialization"] = events["topdown_be_bound.serialization"] / slots
        events["Topdown.Backend_Bound.Resource_Bound.Register"] = events["topdown_be_bound.register"] / slots

        events["Info.Mem_Exec_Blocks.PCT_Loads_with_AdressAliasing"] = events["ld_blocks.4k_alias"] / events["mem_uops_retired.all_loads"]

        events["slots_accounted"] = (events["Topdown.Retiring"] +
                                     events["Topdown.Bad_Speculation.Branch_Mispredicts"] +
                                     events["Topdown.Frontend_Bound.IFetch_Latency"] +
                                     events["Topdown.Frontend_Bound.IFetch_Bandwidth"] +
                                     events["Topdown.Backend_Bound.Core_Bound.Allocation_Restriction"] +
                                     events["Topdown.Backend_Bound.Resource_Bound.Mem_Scheduler"] +
                                     events["Topdown.Backend_Bound.Resource_Bound.Non_Mem_Scheduler"] +
                                     events["Topdown.Backend_Bound.Resource_Bound.Reorder_Buffer"] +
                                     events["Topdown.Backend_Bound.Resource_Bound.Register"] +
                                     events["Topdown.Backend_Bound.Resource_Bound.Serialization"])

compute_extra_metrics(no_opt_bef)
compute_extra_metrics(mlir_bef)
compute_extra_metrics(ains_bef)

with open("no-opt.json", "w") as f, open("pref-mlir-45.json", "w") as g, open("pref-ains-45.json", "w") as h:
    f.write(json.dumps(no_opt_bef, indent=4))
    g.write(json.dumps(mlir_bef, indent=4))
    h.write(json.dumps(ains_bef, indent=4))
