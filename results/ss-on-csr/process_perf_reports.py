import json
import re
from pathlib import Path
import sys

from suite_sparse import SuiteSparse
from typing import Any


def process_perf_report(file: str) -> dict[str, dict[str, Any]]:
    with open(file, "r") as f:
        data = json.load(f)

    mtx_regex = re.compile(r"SuiteSparse (.*)$")

    new_data = {}
    ss = SuiteSparse(Path("."))
    for entry in data:
        mtx = mtx_regex.search(entry["args"]).group(1)
        if mtx not in new_data:
            new_data[mtx] = {}

            # Get some matrix metadata
            new_data[mtx]["url"] = ss.get_info_url(mtx)
            new_data[mtx]["cols"] = int(ss.get_meta(mtx, "num_of_cols"))
            new_data[mtx]["psym"] = float(ss.get_meta(mtx, "psym"))

        # Find event counts
        for e in entry["perf-stat"]:
            new_data[mtx][e["event"].replace("cpu_atom/", "").replace(":u/", "")] = float(e["counter-value"])

    return new_data


def compute_metrics(data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for mtx, events in data.items():
        clks = events["cpu_clk_unhalted.core"]
        slots = 5 * clks

        all_loads = events["mem_uops_retired.all_loads"]
        events["l3_miss_ratio"] = events["mem_load_uops_retired.dram_hit"] / all_loads
        events["l3_hit_ratio"] = events["mem_load_uops_retired.l3_hit"] / all_loads
        events["l2_hit_ratio"] = events["mem_load_uops_retired.l2_hit"] / all_loads
        events["l1_hit_ratio"] = 1 - (events["mem_load_uops_retired.l2_hit"] + events["mem_load_uops_retired.l3_hit"] + events["mem_load_uops_retired.dram_hit"]) / all_loads

        mem_bound_stalls = events["mem_bound_stalls.load"]
        events["Info.Bottleneck.PCT_Load_Miss_Bound_Cycles"] = mem_bound_stalls / clks
        events["Info.Load_Miss_Bound.PCT_LoadMissBound_with_L2Hit"] = events["mem_bound_stalls.load_l2_hit"] / mem_bound_stalls
        events["Info.Load_Miss_Bound.PCT_LoadMissBound_with_L3Hit"] = events["mem_bound_stalls.load_llc_hit"] / mem_bound_stalls
        events["Info.Load_Miss_Bound.PCT_LoadMissBound_with_L3Miss"] = events["mem_bound_stalls.load_dram_hit"] / mem_bound_stalls

        ld_head_any_at_ret = events["ld_head.any_at_ret"]
        events["Info.Bottleneck.PCT_Mem_Exec_Bound_Cycles"] = ld_head_any_at_ret / clks
        events["Info.Mem_Exec_Bound.PCT_LoadHead_with_L1_miss"] = events["ld_head.l1_miss_at_ret"] / ld_head_any_at_ret

        events["Topdown.Frontend_Bound.IFetch_Latency"] = events["topdown_fe_bound.frontend_latency"] / slots
        events["Topdown.Frontend_Bound.IFetch_Bandwidth"] = events["topdown_fe_bound.frontend_bandwidth"] / slots
        events["Topdown.Bad_Speculation.Branch_Mispredicts"] = events["topdown_bad_speculation.mispredict"] / slots
        events["Topdown.Backend_Bound.Core_Bound.Allocation_Restriction"] = events["topdown_be_bound.alloc_restrictions"] / slots
        events["Topdown.Backend_Bound.Resource_Bound.Mem_Scheduler"] = events["topdown_be_bound.mem_scheduler"] / slots
        events["Topdown.Backend_Bound.Resource_Bound.Non_Mem_Scheduler"] = events["topdown_be_bound.non_mem_scheduler"] / slots
        events["Topdown.Backend_Bound.Resource_Bound.Reorder_Buffer"] = events["topdown_be_bound.reorder_buffer"] / slots
        events["Topdown.Retiring"] = events["topdown_retiring.all"] / slots

        events["slots_accounted"] = (events["Topdown.Retiring"] + events["Topdown.Bad_Speculation.Branch_Mispredicts"] +
                                     events["Topdown.Frontend_Bound.IFetch_Latency"] +
                                     events["Topdown.Frontend_Bound.IFetch_Bandwidth"] +
                                     events["Topdown.Backend_Bound.Core_Bound.Allocation_Restriction"] +
                                     events["Topdown.Backend_Bound.Resource_Bound.Mem_Scheduler"] +
                                     events["Topdown.Backend_Bound.Resource_Bound.Non_Mem_Scheduler"] +
                                     events["Topdown.Backend_Bound.Resource_Bound.Reorder_Buffer"])

        events["Info.Buffer_Stalls.PCT_Load_Buffer_Stall_Cycles"] = events["mem_scheduler_block.ld_buf"] / clks
        events["Info.Buffer_Stalls.PCT_Mem_RSV_Stall_Cycles"] = events["mem_scheduler_block.rsv"] / clks

    return dict(sorted(data.items(), key=lambda e: e[1]["Info.Bottleneck.PCT_Load_Miss_Bound_Cycles"], reverse=True))


if __name__ == "__main__":
    data = compute_metrics(process_perf_report(sys.argv[1]))
    with open(sys.argv[1] + ".processed.json", "w") as f:
        f.write(json.dumps(data, indent=4))

