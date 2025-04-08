import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys


def parse_size(size_str):
    val, unit = size_str.strip().split()
    val = float(val)
    unit = unit.upper()
    if unit == "KB":
        return val * 1024
    elif unit == "MB":
        return val * 1024**2
    elif unit == "GB":
        return val * 1024**3
    elif unit == "BYTES":
        return val
    else:
        raise ValueError(f"Unknown unit: {unit}")


if __name__ == "__main__":
    rep = Path(sys.argv[1])
    assert Path(rep).exists(), f"{rep} does not exist"

    with open(rep, "r") as b:
        data = json.load(b)

    densities = []
    nnz_list = []
    mat_sizes = []
    speedups = []

    for _, info in data.items():
        density = info["density"]
        nnz = int(info["num_of_entries"])
        mat_size = parse_size(info["mat_size"])
        speedup = list(info["speed-ups"].values())[0]

        densities.append(density)
        nnz_list.append(nnz)
        mat_sizes.append(mat_size)
        speedups.append(speedup)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].scatter(-np.log10(densities), speedups)
    axs[0].set_xlabel('-log10(density)')
    axs[0].set_ylabel('Speedup')
    axs[0].set_title('Speedup vs Sparsity')
    axs[0].grid(True)

    axs[1].scatter(np.log10(nnz_list), speedups)
    axs[1].set_xlabel('log10(num_of_entries)')
    axs[1].set_ylabel('Speedup')
    axs[1].set_title('Speedup vs NNZ')
    axs[1].grid(True)

    axs[2].scatter(np.log10(mat_sizes), speedups)
    axs[2].set_xlabel('log10(mat_size in bytes)')
    axs[2].set_ylabel('Speedup')
    axs[2].set_title('Speedup vs Matrix Size (bytes)')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
