import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.serif'] = ['Linux Libertine O']
plt.rcParams['font.family'] = 'serif'

with open("vector-sizes.json", "r") as f:
    vector_sizes = json.load(f)

# Convert to sorted array
sizes = np.array(sorted(vector_sizes.values()))
cdf = np.arange(1, len(sizes) + 1) / len(sizes)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(sizes / 1024, cdf * 100, linewidth=2)  # convert to KB for readability
plt.xscale('log')
plt.xlabel('Vector Size (KB)')
plt.ylabel('Matrices (%) ≤ x')
plt.title('CDF of Vector Sizes in SpMV')

# Sizes in KB for x-axis
cache_levels_kb = {
    "L1": (32, "red", "32KB"),
    "L2": (2048, "green", "2MB"),
    "L3": (30720, "blue", "30MB")
}

# Find CDF y-values where each cache size intersects the curve
for label, (size_kb, color, size_str) in cache_levels_kb.items():
    plt.axvline(x=size_kb, color=color, linestyle='--', linewidth=1)
    plt.text(size_kb * 1.2, 5, f'{label} ({size_str})', color=color, rotation=90, va='bottom')

    # Find index of largest vector size ≤ cache size
    idx = np.searchsorted(sizes / 1024, size_kb, side='right') - 1
    if 0 <= idx < len(cdf):
        y_val = cdf[idx] * 100
        # Draw horizontal line
        plt.hlines(y_val, 0, size_kb, color=color, linestyle='--', linewidth=1)
        # Draw dot at intersection
        plt.scatter([size_kb], [y_val], color=color, zorder=5)
        # Add text
        plt.text(size_kb * 0.8, y_val + 1, f'{y_val:.1f}%', ha='right', fontsize=8, color=color)

plt.grid(True, linewidth=0.5, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("cdf-vector-sizes.pdf")
plt.show()
