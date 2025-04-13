import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    rep = Path(sys.argv[1])
    assert rep.exists(), f"{rep} does not exist"
    with open(rep, "r") as f:
        data = json.load(f)

    mpki_vals = [v["mpki"] for v in data.values()]

    # Define explicit bin edges
    bins = np.arange(0, 20.001, 0.001)  # from 0 to 20 in 0.001 steps

    plt.hist(mpki_vals, bins=bins)
    plt.xlabel("MPKI")
    plt.ylabel("Frequency")
    plt.title("Histogram of MPKI values")
    plt.show()
