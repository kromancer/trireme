from argparse import Namespace
from pathlib import Path
from platform import system
import re
from subprocess import run, PIPE
from typing import List

import numpy as np

from hwpref_controller import HwprefController
from input_manager import InputManager
from prof import profile_cmd
from report_manager import  ReportManager


if system() == "Linux":
    from ramdisk_linux import RAMDisk
else:
    assert system() == "Darwin", "Unsupported system!"
    from ramdisk_macos import RAMDisk


def flush_cache(cache_size=100 * 1024 * 1024):
    a = np.arange(cache_size, dtype=np.uint8)
    a.sum()  # forces reading the array


def run_with_aot(args: Namespace, exe: Path, res: np.ndarray, nnz: int, sp_mat_buffs: List[np.array],
                 dense_mat: np.ndarray, exp_out: np.ndarray, in_man: InputManager, rep_man: ReportManager,):

    with (RAMDisk(args, in_man, dense_mat, *sp_mat_buffs, res) as ramdisk, HwprefController(args)):
        cmd = [str(exe), str(args.i), str(args.j), str(args.k), str(nnz)] + ramdisk.buffer_paths
        if args.action == "profile":
            profile_cmd(args, cmd, rep_man)
            if args.check_output:
                assert np.allclose(exp_out, ramdisk.buffers[-1]), "Wrong output!"
        else:
            exec_times = []
            for _ in range(args.repetitions):
                result = run(cmd, check=True, stdout=PIPE, stderr=PIPE, text=True)

                if args.check_output:
                    assert np.allclose(exp_out, ramdisk.buffers[-1]), "Wrong output!"

                ramdisk.reset_res_buff()
                flush_cache()

                match = re.search(r"Exec time: ([0-9.]+)s", result.stdout)
                assert match is not None, "Execution time not found in the output."
                exec_times.append(float(match.group(1)))

            rep_man.log_execution_times_secs(exec_times)
