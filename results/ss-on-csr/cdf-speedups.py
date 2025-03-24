import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.serif'] = ['Linux Libertine O']
plt.rcParams['font.family'] = 'serif'

with open("pref-mlir-45.json", "r") as f, open("pref-ains-45.json", "r") as g:
    data_mlir = json.load(f)
    data_ains = json.load(g)

n = len(data_mlir)
assert n == len(data_ains)
cdf = np.arange(1, n + 1) / n

speedups = np.array(sorted(
    v["speed-up-no-opt"] for v in data_mlir.values()
))

speedups_ains = np.array(sorted(
    v["speed-up-no-opt"] for v in data_ains.values()
))

# Plot
plt.figure(figsize=(8, 5))
plt.xscale('log')
plt.plot(speedups, cdf * 100, label='Speed-up', linewidth=2)
plt.plot(speedups_ains, cdf * 100, label='Speed-up ains', linewidth=2)

# Find the index of the last value ≤ 1
threshold = 1
idx = np.searchsorted(speedups, threshold, side='right') - 1
idx_ains = np.searchsorted(speedups_ains, threshold, side='right') - 1
y_val = cdf[idx] * 100  # Convert to percent for plotting
y_val_ains = cdf[idx_ains] * 100

# Add vertical line for x1 speedup
plt.axvline(x=threshold, color='grey', linestyle='--', linewidth=1)
plt.text(threshold + 0.01, 50, 'x1', color='grey', rotation=90, va='center')

# Add horizontal marker
plt.plot([0, threshold], [y_val, y_val], color='blue', linestyle='--', linewidth=1)
plt.scatter([threshold], [y_val], color='blue')
plt.text(threshold - 0.2, y_val - 6, f'{y_val:.1f}%', color='blue', va='bottom')

plt.plot([0, threshold], [y_val_ains, y_val_ains], color='orange', linestyle='--', linewidth=1)
plt.scatter([threshold], [y_val_ains], color='orange')
plt.text(threshold - 0.2, y_val_ains + 2.5, f'{y_val_ains:.1f}%', color='orange', va='bottom')

# Compute maxima
max_mlir = speedups[-1]
max_ains = speedups_ains[-1]

# Draw vertical lines for each maximum
plt.axvline(x=max_mlir, color='blue', linestyle=':', linewidth=1)
plt.text(max_mlir + 0.02, 50, f'x{max_mlir:.2f}', color='blue', rotation=90, va='center')

plt.axvline(x=max_ains, color='orange', linestyle=':', linewidth=1)
plt.text(max_ains + 0.02, 50, f'x{max_ains:.2f}', color='orange', rotation=90, va='center')

plt.xlabel('Speed-up Factor (log scale)')
plt.ylabel('Matrices (%) ≤ x')
plt.title('CDF of SpMV Speed-ups on SuiteSparse (Single-Threaded)')
plt.legend(bbox_to_anchor=(0.55, 0.55))
plt.grid(True, linewidth=0.5, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("cdf-speedups.pdf")
plt.show()

