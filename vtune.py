from csv import DictReader
from io import StringIO
from multiprocessing import shared_memory
from subprocess import run
import numpy as np
from pathlib import Path
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import scipy.sparse as sp

from common import append_result_to_db, is_in_path, read_config


def gen_and_store_reports() -> None:

    reports = [
        {
            "args": ["hw-events", "-format=csv", "-csv-delimiter=comma", "-group-by=source-line"],
            "output": "vtune-hw-events.csv"
        },
        {
            "args": ["summary", "-format=text"],
            "output": "vtune-summary.txt"
        }
    ]

    db_entry = {}
    for report in reports:
        vtune_cmd = ["vtune", "-report"] + report["args"] + ["-report-output", report["output"]]
        run(vtune_cmd, check=True)

        with open(report["output"], "r") as f:
            db_entry[report["output"]] = f.read()

    append_result_to_db(db_entry)


def plot_observed_max_bandwidth(logs: List[Dict], series: Dict) -> None:
    bw = []
    for log in logs:
        # Regular expression to find the line starting with 'DRAM, GB/sec'
        match = re.search(r'DRAM, GB/sec\s+(\d+)\s+([\d.]+)', log["vtune-summary-txt"])

        if match:
            bw.append(float(match.group(2)))

    x_values = list(range(series['x_start'], series['x_start'] + len(bw)))
    plt.plot(x_values, bw, label=series['label'])


def plot_event(logs: List[Dict], series: Dict) -> None:
    event_counts = []
    for log in logs:
        csv_file = StringIO(log["vtune-hw-events.csv"])
        reader = DictReader(csv_file)

        event = "Hardware Event Count:" + series["event"]
        assert event in reader.fieldnames, f"'{event}' is not a valid column header"

        count = 0
        for row in reader:
            try:
                if "source_line" in series:
                    if int(row["Source Line"]) == series["source_line"]:
                        count = int(row[event])
                        break
                else:
                    count += int(row[event])
            except ValueError:  # Handles non-integer and missing values gracefully
                continue
        event_counts.append(count)

    x_values = list(range(series['x_start'], series['x_start'] + len(event_counts)))
    plt.plot(x_values, event_counts, label=series['label'])


def profile_spmv_with_vtune(exe: Path, mat: sp.csr_array, vec: np.ndarray, vtune_config: str) -> None:

    assert is_in_path("vtune")

    # copy vec to a shared mem block
    vec_shm = shared_memory.SharedMemory(create=True, size=vec.nbytes)
    shared_vec = np.ndarray(vec.shape, dtype=vec.dtype, buffer=vec_shm.buf)
    np.copyto(shared_vec, vec)

    # copy mat.data
    mat_data_shm = shared_memory.SharedMemory(create=True, size=mat.data.nbytes)
    shared_mat_data = np.ndarray(mat.data.shape, dtype=mat.data.dtype, buffer=mat_data_shm.buf)
    np.copyto(shared_mat_data, mat.data)

    # copy mat.indices
    mat_indices_shm = shared_memory.SharedMemory(create=True, size=mat.indices.nbytes)
    shared_mat_indices = np.ndarray(mat.indices.shape, dtype=mat.indices.dtype, buffer=mat_indices_shm.buf)
    np.copyto(shared_mat_indices, mat.indices)

    # copy mat.indptr
    mat_indptr_shm = shared_memory.SharedMemory(create=True, size=mat.indptr.nbytes)
    shared_mat_indptr = np.ndarray(mat.indptr.shape, dtype=mat.indptr.dtype, buffer=mat_indptr_shm.buf)
    np.copyto(shared_mat_indptr, mat.indptr)

    # create res buffer
    num_of_rows = mat.shape[0]
    all_zeroes = np.zeros(num_of_rows, dtype=mat.data.dtype)
    res_shm = shared_memory.SharedMemory(create=True, size=all_zeroes.nbytes)
    res = np.ndarray(all_zeroes.shape, dtype=mat.data.dtype, buffer=res_shm.buf)
    np.copyto(res, all_zeroes)

    vtune_cmd = ["vtune"] + read_config("vtune-config.json", vtune_config) + ["--"]
    spmv_cmd = [exe, str(num_of_rows), "/" + vec_shm.name, "/" + mat_data_shm.name, "/" + mat_indptr_shm.name,
                "/" + mat_indices_shm.name, "/" + res_shm.name]

    try:
        run(vtune_cmd + spmv_cmd, check=True)
    finally:
        vec_shm.unlink()
        mat_data_shm.unlink()
        mat_indptr_shm.unlink()
        mat_indices_shm.unlink()
        res_shm.unlink()

    gen_and_store_reports()

    expected = mat.dot(vec)
    assert np.allclose(res, expected), "Wrong result!"
