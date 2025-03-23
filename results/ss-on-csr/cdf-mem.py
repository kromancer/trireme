import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.serif'] = ['Linux Libertine O']
plt.rcParams['font.family'] = 'serif'

with open("consolidated-no-opt.json", "r") as f:
    data = json.load(f)

# Extract and sort values
load_miss = np.array(sorted(
    v["Info.Bottleneck.PCT_Load_Miss_Bound_Cycles"] for v in data.values()
))
x = np.array(sorted(
    (v["mem_load_uops_retired.l2_hit"] + v["mem_load_uops_retired.l3_hit"]) /
    (v["mem_load_uops_retired.l2_hit"] + v["mem_load_uops_retired.l3_hit"] + v["mem_load_uops_retired.dram_hit"])
    for v in data.values()
))

n = len(load_miss)
assert len(x) == len(load_miss)

# Compute CDFs
cdf = np.arange(1, n + 1) / n

# Plot
plt.figure(figsize=(8, 5))
plt.plot(load_miss * 100, cdf * 100, label='%Cycles retirement is stalled due to an L1 miss', linewidth=3)
plt.plot(x * 100, cdf * 100, label='%L1 misses that hit L2/L3 (structure score)', linewidth=3)

# Find the index of the last value ≤ 0.9
threshold = 0.9315
idx = np.searchsorted(x, threshold, side='right') - 1
y_val = cdf[idx] * 100  # Convert to percent for plotting

# Add vertical line for 90% structure score
plt.axvline(x=threshold * 100, color='red', linestyle='--', linewidth=1)
plt.text(threshold * 100 - 3, 50, '93% structure score', color='red', rotation=90, va='center')

# Add horizontal marker
plt.plot([0, threshold * 100], [y_val, y_val], color='red', linestyle='--', linewidth=1)
plt.scatter([threshold * 100], [y_val], color='red')
plt.text(threshold * 100 + 3, y_val, f'{y_val:.1f}%', color='red', va='bottom')

# For the load_miss CDF (also in [0,1] range, so 90% = 0.9)
threshold = 0.252
idx_miss = np.searchsorted(load_miss, threshold, side='right') - 1
y_val_miss = cdf[idx_miss] * 100  # Convert to %

# Add vertical line
plt.axvline(x=threshold * 100, color='blue', linestyle='--', linewidth=1)
plt.text(threshold * 100 + 2, 50, '25% Stalled Cycles', color='blue', rotation=90, va='center')

# Add horizontal marker
plt.plot([0, threshold * 100], [y_val_miss, y_val_miss], color='blue', linestyle='--', linewidth=1)
plt.scatter([threshold * 100], [y_val_miss], color='blue')
plt.text(threshold * 100 + 3, y_val_miss, f'{y_val_miss:.1f}%', color='blue', va='bottom')

plt.xlabel('x (%)')
plt.ylabel('Percentage of Matrices ≤ x')
plt.title('SpMV-Baseline on SuiteSparse, single threaded')
plt.legend(bbox_to_anchor=(0.55, 0.55))
plt.grid(True, linewidth=0.5, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("cdf-mem.pdf")
plt.show()
