from common import json_load_and_backup, json_load
import json
from statistics import mean
import sys


if __name__ == '__main__':
    data = json_load_and_backup(sys.argv[1])

    for mtx, v in data.items():
        mean_instr = mean(v["inst_retired.any"])
        mean_l3_misses = mean(v["mem_load_uops_retired.dram_hit"])
        v["mpki"] = mean_l3_misses * 1000 / mean_instr
        mean_l2_misses = mean(v["mem_load_uops_retired.l3_hit"] + v["mem_load_uops_retired.dram_hit"])
        v["l2mpki"] = mean_l2_misses * 1000 / mean_instr

    with open(sys.argv[1], "w") as f:
        f.write(json.dumps(data, indent=4))
