import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.serif'] = ['Linux Libertine O']
plt.rcParams['font.family'] = 'serif'

with open("./spmm-isolate/1GB/pref-mlir.json", "r") as f:
    data_mlir = json.load(f)

n = len(data_mlir)
cdf = np.arange(1, n + 1) / n

speedups = np.array(sorted(
    v["speed-ups"]["/Users/ioanniss/trireme/results/spmm-isolate/1GB/no-opt.json"] for v in data_mlir.values()
))

# Plot
plt.figure(figsize=(8, 5))
plt.xscale('log')
plt.plot(speedups, cdf * 100, label='Speedup MLIR-Pref', linewidth=2)

# Find the index of the last value ≤ 1
threshold = 1
idx = np.searchsorted(speedups, threshold, side='right') - 1
y_val = cdf[idx] * 100  # Convert to percent for plotting


# Add vertical line for x1 speedup
plt.axvline(x=threshold, color='grey', linestyle='--', linewidth=1)
plt.text(threshold + 0.01, 50, '1x', color='grey', rotation=90, va='center')

# Add horizontal marker
plt.plot([0, threshold], [y_val, y_val], color='blue', linestyle='--', linewidth=1)
plt.scatter([threshold], [y_val], color='blue')
plt.text(threshold - 0.2, y_val - 6, f'{y_val:.1f}%', color='blue', va='bottom')


# Compute maxima
max_mlir = speedups[-1]

# Draw vertical lines for each maximum
plt.axvline(x=max_mlir, color='blue', linestyle=':', linewidth=1)
plt.text(max_mlir + 0.02, 50, f'{max_mlir:.2f}x', color='blue', rotation=90, va='center')


plt.xlabel('Speedup Factor (log scale)')
plt.ylabel('Matrices (%) ≤ x')
plt.title('CDF of SpMM Speedups on SuiteSparse (Single-Threaded)')
plt.legend(bbox_to_anchor=(0.55, 0.55))
plt.grid(True, linewidth=0.5, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("cdf-speedups.pdf")
plt.show()

