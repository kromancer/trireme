import json
from csv import DictReader
from io import StringIO
from pathlib import Path
import re
from statistics import harmonic_mean
from typing import Dict, List, Optional

from matplotlib import pyplot as plt

from input_manager import InputManager
from suite_sparse import SuiteSparse


def filter_logs(log: Path, args_re: str) -> List[Dict]:
    with open(log, 'r') as f:
        logs = json.load(f)
    filtered = [log for log in logs if re.search(args_re, log['args'])]
    return filtered


def plot_mean_exec_times(logs: List[Dict], series: Dict, normalize: Optional[float]) -> None:
    means = [log['mean_ms'] for log in logs]

    if normalize is not None:
        means = [log['mean_ms'] / normalize for log in logs]

    x_vals = list(range(series['x_start'], series['x_start'] + len(means)))

    # Find the minimum value and its index
    min_value = min(means)
    min_index = means.index(min_value)

    # Calculate the actual x value for the minimum point
    min_x_value = x_vals[min_index]

    # Highlight the minimum point
    plt.scatter(min_x_value, min_value, color='red', s=20, zorder=5)

    plt.plot(x_vals, means, label=series['label'] + f", min({min_x_value:.0f}, {min_value:.3f})")


def plot_mean_exec_times_ss(logs: List[Dict], series: Dict, normalize: Optional[float]) -> None:
    """
    Plot mean exec times for SparseSuite matrices
    X-axis is the number of non-zero elements of the matrix
    """
    m_names = [re.search(r'SuiteSparse\s+(\S+)', log['args']).group(1) for log in logs]
    names_to_nnz = SuiteSparse(InputManager.get_working_dir("SuiteSparse")).get_all_matrix_names_with_nnz()
    nnz = [names_to_nnz[n] for n in m_names]

    means_all = []
    for log in logs:
        try:
            means_all.append(log['mean_ms'])
        except KeyError:
            print(f"mean_ms not present in {log}")
            means_all.append(0)

    work_rates_all = [x / m for x, m in zip(nnz, means_all) if m > 0]
    work_rates_big = [x / m for x, m in zip(nnz, means_all) if x > 10 ** 7 and m > 0]
    ha_all = harmonic_mean(work_rates_all)
    ha_big = harmonic_mean(work_rates_big)

    lbl = series['label'] + f", HA_all: {int(ha_all)}, HA_big: {int(ha_big)}"
    plt.scatter(nnz, means_all, label=lbl, s=5)
    plt.xscale("log")
    plt.yscale("log")


def plot_events_perf_ss(logs: List[Dict], series: Dict, normalize: Optional[float]) -> None:
    """
    Plot perf mon events for SparseSuite matrices
    X-axis is the number of non-zero elements of the matrix
    """
    event_name = series["event"]
    e_pattern = re.compile(
        r"event\s+'{}'.*?# Event count \(approx\.\):\s*(\d+)".format(re.escape(event_name)),
        re.DOTALL
    )

    norm_pattern = None
    if "normalize" in series:
        normalization_event = series["normalize"]
        norm_pattern = re.compile(
            r"event\s+'{}'.*?# Event count \(approx\.\):\s*(\d+)".format(re.escape(normalization_event)),
            re.DOTALL
        )

    e_counts = []
    x_vals = []
    names_to_nnz = SuiteSparse(working_dir=InputManager.get_working_dir("SuiteSparse")).get_all_matrix_names_with_nnz()
    for log in logs:
        if "perf-report" not in log:
            assert log["status"] != "complete"
            print(f"perf-report not in {log['args']}")
            continue
        match = e_pattern.search(log["perf-report"])
        if match:
            e_counts.append(float(match.group(1)))
            m_name = re.search(r'SuiteSparse\s+(\S+)', log["args"]).group(1)
            x_vals.append(names_to_nnz[m_name])

        if norm_pattern is not None:
            match2 = norm_pattern.search(log["perf-report"])
            if match2:
                norm = float(match2.group(1))
                normalized = e_counts.pop() / norm
                assert normalized <= 1
                e_counts.append(normalized)

    plt.scatter(x_vals, e_counts, label=series['label'], s=5)
    plt.xscale("log")
    plt.yscale("log")


def plot_observed_max_bandwidth(logs: List[Dict], series: Dict, normalize: Optional[float]) -> None:
    bw = []
    for log in logs:
        # Regular expression to find the line starting with 'DRAM, GB/sec'
        match = re.search(r'DRAM, GB/sec\s+(\d+)\s+([\d.]+)', log["vtune-summary-txt"])

        if match:
            bw.append(float(match.group(2)))

    x_values = list(range(series['x_start'], series['x_start'] + len(bw)))
    plt.plot(x_values, bw, label=series['label'])


def plot_events_from_vtune(logs: List[Dict], series: Dict, normalize: Optional[float]) -> None:
    event_counts = []
    for log in logs:
        csv_file = StringIO(log["vtune-hw-events.csv"])
        reader = DictReader(csv_file)

        event = "Hardware Event Count:" + series["event"]
        assert event in reader.fieldnames, f"'{event}' is not a valid column header"

        if "normalize" in series:
            norm_event = "Hardware Event Count:" + series["normalize"]
            assert norm_event in reader.fieldnames, f"'{norm_event}' is not a valid column header"

        count = 0
        norm = 0
        for row in reader:
            try:
                if "source_line" in series:
                    if int(row["Source Line"]) == series["source_line"]:
                        count = int(row[event])
                else:
                    count += int(row[event])

                if "normalize" in series:
                    norm += int(row[norm_event])

            except ValueError:  # Handles non-integer and missing values gracefully
                continue

        event_counts.append(round((100 * count) / (1 if norm == 0 else norm), 4))

    print(event_counts)
    x_values = list(range(series['x_start'], series['x_start'] + len(event_counts)))
    plt.plot(x_values, event_counts, label=series['label'])


def main():
    config_file = './plot-config.json'
    with open(config_file, 'r') as file:
        config = json.load(file)

    base_path = Path(config_file).resolve().parent

    normalize = None
    if "normalize" in config:
        log_path = base_path / Path(config["normalize"]['file']).relative_to('.')
        means = filter_logs(log_path, config["normalize"]['args_re'])
        assert len(means) == 1
        normalize = means[0]["mean_ms"]

    for series in config['series']:
        log = base_path / Path(series['file']).relative_to('.')

        logs = filter_logs(log, series['args_re'])

        # plot series based on specified method
        {
            "plot_mean_exec_times": plot_mean_exec_times,
            "plot_observed_max_bandwidth": plot_observed_max_bandwidth,
            "plot_event": plot_events_from_vtune,
            "plot_events_perf_ss": plot_events_perf_ss,
            "plot_mean_exec_times_ss": plot_mean_exec_times_ss
        }[series["plot_method"]](logs, series, normalize)

    if "baselines" in config:
        for baseline in config['baselines']:
            log_path = base_path / Path(baseline['file']).relative_to('.')
            means = filter_logs(log_path, baseline['args_re'])
            assert len(means) == 1
            plt.axhline(y=means[0]["mean_ms"], color=baseline["color"], linestyle=baseline["linestyle"],
                        label=baseline["label"] + f", {means[0]['mean_ms']:.0f}ms")

    plt.xlabel(config['xlabel'])
    plt.ylabel(config['ylabel'])
    plt.title(config['title'], fontsize='small')
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.savefig("fig.pdf")

    plt.show()


if __name__ == "__main__":
    main()
