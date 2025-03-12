import json
import re
from pathlib import Path
import sys

from suite_sparse import SuiteSparse


def process_perf_report(file: str):
    with open(file, "r") as f:
        data = json.load(f)

    e_regexes = {
        "mem_bound_stalls.load": re.compile(r"'cpu_atom/mem_bound_stalls.load/upp'\s*# Event count.*?: (\d*)"),
        "topdown_bad_speculation.mispredict": re.compile(r"'cpu_atom/topdown_bad_speculation.mispredict/upp'\s*# Event count.*?: (\d*)"),
        "mem_load_uops_retired.dram_hit": re.compile(r"'cpu_atom/mem_load_uops_retired.dram_hit/upp'\s*# Event count.*?: (\d*)"),
        "topdown_be_bound.reorder_buffer": re.compile(r"'cpu_atom/topdown_be_bound.reorder_buffer/upp'\s*# Event count.*?: (\d*)"),
        "topdown_be_bound.mem_scheduler": re.compile(r"'cpu_atom/topdown_be_bound.mem_scheduler/upp'\s*# Event count.*?: (\d*)"),
        "cpu_clk_unhalted.core": re.compile(r"'cpu_atom/cpu_clk_unhalted.core/upp'\s*# Event count.*?: (\d*)")
    }
    mtx_regex = re.compile(r"SuiteSparse (.*)$")

    new_data = []
    ss = SuiteSparse(Path("."))
    for entry in data:
        mtx = mtx_regex.search(entry["args"]).group(1)
        new_e = {"mtx": mtx}

        # Get some matrix metadata
        new_e["num_of_cols"] = int(ss.get_meta(mtx, "num_of_cols"))

        # Find event counts
        rep = entry["perf-report"]
        for event, regex in e_regexes.items():
            m = regex.search(rep)
            new_e[event] = int(m.group(1)) if m else 0

        # Compute metrics
        # 1. Info.Bottleneck.Load_Miss_Bound_Cycles
        if new_e["cpu_clk_unhalted.core"] != 0:
            new_e["Info.Bottleneck.Load_Miss_Bound_Cycles"] = new_e["mem_bound_stalls.load"] / new_e["cpu_clk_unhalted.core"]
        else:
            new_e["Info.Bottleneck.Load_Miss_Bound_Cycles"] = 0

        # 2. Backend_Bound.Resource_Bound.Reorder_Buffer
        slots = 5 * new_e["cpu_clk_unhalted.core"]
        if slots != 0:
            new_e["Backend_Bound.Resource_Bound.Reorder_Buffer"] = new_e["topdown_be_bound.reorder_buffer"] / slots
        else:
            new_e["Backend_Bound.Resource_Bound.Reorder_Buffer"] = 0

        # 3. Backend_Bound.Resource_Bound.Mem_Scheduler
        if slots != 0:
            new_e["Backend_Bound.Resource_Bound.Mem_Scheduler"] = new_e["topdown_be_bound.mem_scheduler"] / slots
        else:
            new_e["Backend_Bound.Resource_Bound.Mem_Scheduler"] = 0

        # 4. Bad_Speculation.Branch_Mispredicts
        if slots != 0:
            new_e["Bad_Speculation.Branch_Mispredicts"] = new_e["topdown_bad_speculation.mispredict"] / slots
        else:
            new_e["Bad_Speculation.Branch_Mispredicts"] = 0

        new_data.append(new_e)

    sort_on_load_miss_bound = sorted(new_data, key=lambda e: e["Info.Bottleneck.Load_Miss_Bound_Cycles"], reverse=True)

    with open(file + ".processed.json", "w") as f:
        f.write(json.dumps(sort_on_load_miss_bound, indent=4))


if __name__ == "__main__":
    process_perf_report(sys.argv[1])
