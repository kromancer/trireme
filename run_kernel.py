from argparse import Namespace
from platform import system
import re
from subprocess import run, PIPE
from typing import List, Optional

import numpy as np

from hwpref_controller import HwprefController
from input_manager import InputManager
from prof import profile_cmd
from report_manager import ReportManager


if system() == "Linux":
    from ramdisk_linux import RAMDisk

else:
    assert system() == "Darwin", "Unsupported system!"
    from ramdisk_macos import RAMDisk


def trash_dram_and_flush_cache(size=128 * 1024 * 1024):
    a = np.random.randint(0, 256, size, dtype=np.uint8)
    a.sum()


def run_with_aot(args: Namespace, partial_cmd: List[str], res: np.ndarray, sp_mat_buffs: List[np.array],
                 dense_op: np.ndarray, exp_out: Optional[np.ndarray], in_man: InputManager, rep_man: ReportManager):
    with HwprefController(args):
        if args.action == "profile":
            with RAMDisk(args, in_man, dense_op, *sp_mat_buffs, res) as ramdisk:
                cmd = partial_cmd + ramdisk.buffer_paths
                profile_cmd(args, cmd, rep_man)
                if args.check_output:
                    assert np.allclose(exp_out, ramdisk.buffers[-1]), "Wrong output!"
        else:  # benchmarking
            exec_times = []
            re_time = re.compile(r"Exec time: ([0-9.]+)s")

            # events = ["cpu_clk_unhalted.core", "inst_retired.any", "mem_uops_retired.all_loads", "mem_load_uops_retired.dram_hit"]
            events = ["inst_retired.any", "mem_load_uops_retired.l2_hit", "mem_load_uops_retired.l3_hit", "mem_load_uops_retired.dram_hit"]
            event_re = {}
            for e in events:
                event_re[e] = re.compile(e + ": ([0-9]+)")
            event_counts = {}
            for _ in range(args.repetitions):
                with RAMDisk(args, in_man, dense_op, *sp_mat_buffs, res) as ramdisk:
                    cmd = partial_cmd + ramdisk.buffer_paths
                    result = run(cmd, check=True, stdout=PIPE, stderr=PIPE, text=True)

                    if args.check_output:
                        assert np.allclose(exp_out, ramdisk.buffers[-1]), "Wrong output!"

                    ramdisk.reset_res_buff()
                    trash_dram_and_flush_cache()

                    if system() == "Linux":
                        with open("/proc/sys/vm/drop_caches", "w") as f:
                            f.write("3\n")

                    match = re_time.search(result.stdout)
                    assert match is not None, "Execution time not found in the output."
                    exec_times.append(float(match.group(1)))

                    for e in events:
                        match = event_re[e].search(result.stdout)
                        assert match is not None, f"{e} not found in the output."
                        event_counts.setdefault(e, []).append(int(match.group(1)))

            rep_man.log_execution_times_secs(exec_times)
            rep_man.log_events(event_counts)
