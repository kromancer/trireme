import matplotlib.pyplot as plt
import numpy as np
from common import json_load
import sys

plt.rcParams['font.serif'] = ['Linux Libertine O']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

if __name__ == "__main__":
    data = json_load(sys.argv[1])
    speedup_f = sys.argv[2]
    mpki_f = sys.argv[3]

    # Optional filtering
    if len(sys.argv) == 6:
        filter_field = sys.argv[4]
        filter_value = sys.argv[5]
        data = {k: v for k, v in data.items() if str(v.get(filter_field)) == filter_value}

    x = np.array([d[mpki_f] for d in data.values()])
    y = np.array([d[speedup_f] for d in data.values()])

    coeff = np.polyfit(x, y, 1)
    fit = np.poly1d(coeff)

    y_pred = fit(x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    plt.xscale('log')
    plt.scatter(x, y, color='#1b9e77', alpha=0.5)

    xs = np.linspace(min(x), max(x), 200)
    plt.plot(xs, fit(xs), color='#1b9e77', label=fr'$y = {coeff[0]:.3f}x + {coeff[1]:.3f},\ R^2 = {r_squared:.3f}$')

    plt.xlabel("log(L2 MPKI) on Baseline")
    plt.ylabel("Speedup MLIR-Pref")
    plt.legend()
    plt.savefig("spmv.pdf")
    plt.show()
