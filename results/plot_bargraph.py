import sys
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hmean

from results.compute_work_metric import compute_work_metrics_for_groups

plt.rcParams['font.serif'] = ['Linux Libertine O']
plt.rcParams['font.family'] = 'serif'


# GROUPS = ["SNAP", "GAP", "Sybrandt", "DIMACS10", "Pajek", "Gset", "Mycielski", "Gleich"]
GROUPS = ["SNAP", "GAP", "Sybrandt", "DIMACS10", "Pajek", "Gleich"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Input JSON files (baseline first)")
    parser.add_argument("--labels", nargs="+", help="Labels for the bars (same order as files)")
    return parser.parse_args()


def main():
    args = parse_args()
    filenames = args.files
    labels = args.labels if args.labels else [
        f.split('/')[-1].replace('.json', '') for f in filenames
    ]

    if len(labels) != len(filenames):
        print("Number of labels must match number of files")
        sys.exit(1)

    all_data = [json.load(open(f)) for f in filenames]

    heights = []
    for data in all_data:
        _, metrics = zip(*(compute_work_metrics_for_groups(data, [g]) for g in GROUPS))
        heights.append(metrics)

    baseline = np.array(heights[0])
    normalized = np.array(heights) / baseline
    normalized = normalized.tolist()

    # Add harmonic mean group
    baseline_hm = hmean(heights[0])
    hm_ratios = [hmean(h) / baseline_hm for h in heights]
    for i in range(len(heights)):
        heights[i] = list(heights[i]) + [hmean(heights[i])]
        normalized[i] = normalized[i] + [hm_ratios[i]]

    GROUPS.append(r"H$_{\mathrm{throughput}}$")

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(GROUPS))
    width = 0.8 / len(filenames)
    for i, label in enumerate(labels):
        bars = ax.bar(x + i * width, normalized[i], width, label=label)
        if i > 0:  # skip labeling baseline bars
            for j, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{normalized[i][j]:.2f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width * (len(filenames) - 1) / 2)
    ax.set_xticklabels(GROUPS, rotation=45, ha='right')
    ax.set_ylabel("Normalized Throughput")
    ax.set_title("Throughput Comparison per Matrix Group")
    ax.set_ylim(0, 1.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig("bargraph.pdf")
    plt.show()


if __name__ == "__main__":
    main()
