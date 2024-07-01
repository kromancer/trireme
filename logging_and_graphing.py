import csv
import json
from pathlib import Path
import re
from statistics import mean, median, stdev
from typing import Dict, List

import matplotlib.pyplot as plt

from common import append_result_to_db
from vtune import plot_observed_max_bandwidth, plot_event


def filtered_by_median(data) -> List[int]:
    r""" Temporary hack until we can guarantee that there is no context switching in the core running the kernel"""
    median_value = median(data)

    filtered = [x for x in data if x <= 1.5 * median_value]
    outliers = [x for x in data if x not in filtered]

    if outliers:
        print("Outliers:", outliers)

    return filtered


def log_execution_times_secs(etimes_s: List[float]):
    log_execution_times_ns([int(t * 1e9) for t in etimes_s])


def log_execution_times_ns(etimes_ns: List[int]):

    filtered = filtered_by_median(etimes_ns)

    if len(filtered) <= 2:
        print(f"Number of measurements, after filtering is < 2, skipping logging")
        print(f"All entries: {etimes_ns}")
        return

    m = round(mean(filtered) / 1000000, 3)
    std_dev = round(stdev(filtered) / 1000000, 3)
    cv = round(std_dev / m, 3) if m != 0 else 0
    print(f"mean execution time: {m} ms")
    print(f"std dev: {std_dev} ms, CV: {cv * 100} %")
    append_result_to_db({
        'exec_times_ns': etimes_ns,
        'filtered': filtered,
        'mean_ms': m,
        'std_dev': std_dev,
        'cv': cv})


def filter_logs(log: Path, args_re: str) -> List[Dict]:
    with open(log, 'r') as f:
        logs = json.load(f)
    filtered = [log for log in logs if re.search(args_re, log['args'])]
    return filtered


def save_hw_event_report_to_csv(log: str, args_re: str) -> Path:
    log = Path(log)
    assert log.exists()

    entry = filter_logs(log=log, args_re=args_re)
    assert len(entry) == 1
    entry = entry[0]

    with open("hw-events.csv", "w") as f:
        f.write(entry["vtune-hw-events.csv"])


def plot_mean_exec_times(logs: List[Dict], series: Dict) -> None:
    means = [log['mean_ms'] for log in logs]
    x_values = list(range(series['x_start'], series['x_start'] + len(means)))

    # Find the minimum value and its index
    min_value = min(means)
    min_index = means.index(min_value)

    # Calculate the actual x value for the minimum point
    min_x_value = x_values[min_index]

    # Highlight the minimum point
    plt.scatter(min_x_value, min_value, color='red', s=20, zorder=5)

    plt.plot(x_values, means, label=series['label'] + f", min({min_x_value:.0f}, {min_value:.0f})")


def main():
    config_file = './graph-config.json'
    with open(config_file, 'r') as file:
        config = json.load(file)

    base_path = Path(config_file).resolve().parent

    for series in config['series']:
        log = base_path / Path(series['file']).relative_to('.')

        logs = filter_logs(log, series['args_re'])

        # plot series based on specified method
        {
            "plot_mean_exec_times": plot_mean_exec_times,
            "plot_observed_max_bandwidth": plot_observed_max_bandwidth,
            "plot_event": plot_event
        }[series["plot_method"]](logs, series)

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
