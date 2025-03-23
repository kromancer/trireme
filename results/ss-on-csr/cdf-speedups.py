import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.serif'] = ['Linux Libertine O']
plt.rcParams['font.family'] = 'serif'

with open("pref-mlir-45.json", "r") as f:
    data = json.load(f)

n = len(data)
cdf = np.arange(1, n + 1) / n

speedups = np.array(sorted(
    v["speed-up-no-opt"] for v in data.values()
))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(speedups, cdf * 100, label='Speed-up', linewidth=3)

# Find the index of the last value ≤ 1
threshold = 1
idx = np.searchsorted(speedups, threshold, side='right') - 1
y_val = cdf[idx] * 100  # Convert to percent for plotting

# Add vertical line for 90% structure score
plt.axvline(x=threshold, color='blue', linestyle='--', linewidth=1)
plt.text(threshold + 0.01, 50, 'x1', color='blue', rotation=90, va='center')

# Add horizontal marker
plt.plot([0, threshold], [y_val, y_val], color='blue', linestyle='--', linewidth=1)
plt.scatter([threshold], [y_val], color='blue')
plt.text(threshold + 0.1, y_val, f'{y_val:.1f}%', color='blue', va='bottom')

plt.xlabel('speed-up')
plt.ylabel('Percentage of Matrices ≤ x')
plt.title('SpMV MLIR-Pref on SuiteSparse, single threaded')
plt.legend(bbox_to_anchor=(0.55, 0.55))
plt.grid(True, linewidth=0.5, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("cdf-speedups.pdf")
plt.show()

