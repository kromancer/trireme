from datetime import datetime
import json
from pathlib import Path
import re
from socket import gethostname
from statistics import mean, median, stdev
import sys
from typing import Tuple, List
from git import Repo
import matplotlib.pyplot as plt


def append_entry_to_json(new_entry, file_path=None):
    if file_path is None:
        # Set default file_path to 'logs.json' in the current script's directory
        file_path = Path(__file__).resolve().parent / 'logs.json'
    else:
        file_path = Path(file_path)

    # Get the absolute path of the file
    abs_file_path = file_path.resolve()

    # Attempt to open the file, create a new one if it does not exist
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
        print(f"No existing file found. A new file will be created at: {abs_file_path}")

    # Check if data is a list, if not, initialize as a list
    if not isinstance(data, list):
        data = []

    # Append the new entry
    data.append(new_entry)

    # Write the updated list back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def get_git_commit_hash():
    # Convert script_path to a Path object to find the repo's root directory
    repo_path = Path(__file__).resolve().parent

    # Initialize a Repo object using the script's directory
    repo = Repo(repo_path, search_parent_directories=True)

    # Get the current commit hash
    commit_hash = repo.head.commit.hexsha

    return commit_hash


def filtered_by_median(data) -> List[int]:
    r""" Temporary hack until we can guarantee that there is no context switching in the core running the kernel"""
    median_value = median(data)

    filtered = [x for x in data if x <= 1.5 * median_value]
    outliers = [x for x in data if x not in filtered]

    if outliers:
        print("Outliers:", outliers)

    return filtered


def log_execution_times(etimes_ns: List[int]):
    if len(etimes_ns) <= 2:
        return

    filtered = filtered_by_median(etimes_ns)
    m = round(mean(filtered) / 1000000, 3)
    std_dev = round(stdev(filtered) / 1000000, 3)
    cv = round(std_dev / m, 3)
    print(f"mean execution time: {m} ms")
    print(f"std dev: {std_dev} ms, CV: {cv} %")
    append_entry_to_json({'args': ' '.join(sys.argv),
                          'time': str(datetime.now()),
                          'host': gethostname(),
                          'git-hash': get_git_commit_hash(),
                          'exec_times_ns': etimes_ns,
                          'filtered': filtered,
                          'mean_ms': m,
                          'std_dev': std_dev,
                          'cv': cv})


def parse_logs(file_path: str) -> Tuple[List, List, List]:
    with open(file_path, 'r') as file:
        logs = json.load(file)

    l0, l2, l3 = [], [], []
    for log in logs:
        args = log['args']
        mean_ms = log['mean_ms']

        # Check the -l value and append mean_ms to the respective list
        if '-l 0' in args:
            l0.append(mean_ms)
        elif '-l 2' in args:
            l2.append(mean_ms)
        elif '-l 3' in args:
            l3.append(mean_ms)

    return l0, l2, l3


def filter_logs(log: Path, args_re: str) -> List[int]:
    with open(log, 'r') as f:
        logs = json.load(f)
    filtered_means = [log['mean_ms'] for log in logs if re.match(args_re, log['args'])]
    return filtered_means


if __name__ == "__main__":
    config_file = './graph-config.json'
    with open(config_file, 'r') as file:
        config = json.load(file)

    base_path = Path(config_file).resolve().parent

    for series in config['series']:
        log_path = base_path / Path(series['file']).relative_to('.')
        means = filter_logs(log_path, series['args_re'])
        x_values = list(range(series['x_start'], series['x_start'] + len(means)))
        plt.plot(x_values, means, label=series['label'])

    # Plot a horizontal line for the Baseline section
    plt.axhline(y=7.283, color='r', linestyle='-', label='Baseline')

    plt.xlabel(config['xlabel'])
    plt.ylabel(config['ylabel'])
    plt.title(config['title'])
    plt.legend()
    plt.grid(True)
    plt.show()
