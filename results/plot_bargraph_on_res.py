import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hmean

from common import json_load

plt.rcParams['font.serif'] = ['Linux Libertine O']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20


GROUPS = ["Sybrandt", "GAP", "Gleich", "SNAP", "Pajek", "MAWI", "DIMACS10"]

# Baseline label + color mapping
BASELINE_META = {
    "nnz_per_ms":           ("Baseline", "#7570b3"),
    "nnz_per_ms_ains-pref": ("Ains \& Jones", "#d95f02"),
}


def main():
    data = json_load(sys.argv[1])
    baseline_key = sys.argv[2] if len(sys.argv) > 2 else "nnz_per_ms"

    if baseline_key not in BASELINE_META:
        raise ValueError(f"Unsupported baseline: {baseline_key}")

    baseline_label, baseline_color = BASELINE_META[baseline_key]

    throughputs_base = {}
    throughputs_mlir = {}
    for mtx, v in data.items():
        if v["group"] in GROUPS:
            throughputs_base.setdefault(v["group"], []).append(v[baseline_key])
            throughputs_mlir.setdefault(v["group"], []).append(v["nnz_per_ms_mlir-pref"])

    hmean_groups_base = {}
    hmean_groups_mlir = {}
    for g in GROUPS:
        hmean_groups_base[g] = hmean(throughputs_base[g])
        hmean_groups_mlir[g] = hmean(throughputs_mlir[g])

    base_normalized = {}
    mlir_normalized = {}
    for g, v in hmean_groups_mlir.items():
        base_normalized[g] = 1
        mlir_normalized[g] = v / hmean_groups_base[g]

    base_normalized["all"] = 1
    mlir_normalized["all"] = hmean(list(hmean_groups_mlir.values())) / hmean(list(hmean_groups_base.values()))

    rest_nnz_per_ms_base = []
    rest_nnz_per_ms_mlir = []
    for mtx, v in data.items():
        if v["group"] not in GROUPS:
            rest_nnz_per_ms_base.append(v[baseline_key])
            rest_nnz_per_ms_mlir.append(v["nnz_per_ms_mlir-pref"])

    base_normalized["ews_rest"] = 1
    mlir_normalized["ews_rest"] = hmean(rest_nnz_per_ms_mlir) / hmean(rest_nnz_per_ms_base)

    group_keys = GROUPS + ["all", "ews_rest"]
    group_labels = GROUPS + [r"Selected", r"Others"]

    base_vals = [base_normalized[k] for k in group_keys]
    mlir_vals = [mlir_normalized[k] for k in group_keys]

    x = np.arange(len(group_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.bar(x - width / 2, base_vals, width, label=baseline_label, color=baseline_color)
    ax.bar(x + width / 2, mlir_vals, width, label='MLIR-Pref', color='#1b9e77')

    for i, (b, m) in enumerate(zip(base_vals, mlir_vals)):
        speedup = m / b
        ax.text(x[i] + width / 2 + 0.3 * width, m + 0.03, f'{speedup:.2f}×', ha='center', va='bottom', fontsize=16)

    ax.set_yticks([0, 1, 2])
    ax.set_ylabel('Equal-Work harmonic mean Speedup (EWS)')
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=45, ha='right')
    ax.legend()

    plt.savefig("spmv_bargraph.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
