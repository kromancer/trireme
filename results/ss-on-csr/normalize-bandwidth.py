import json
from pathlib import Path

from suite_sparse import SuiteSparse

with open("bandwidth-results.json", "r") as f:
    data = json.load(f)


ss = SuiteSparse(Path("."))
new_data = {}
for mtx, (l, u) in data.items():
    rows = int(ss.get_meta(mtx, "num_of_rows"))
    cols = int(ss.get_meta(mtx, "num_of_cols"))
    try:
        low = l / (rows - 1)
    except ZeroDivisionError:
        low = 0
    try:
        upper = u / (cols - 1)
    except ZeroDivisionError:
        upper = 0
    new_data[mtx] = max(low, upper)


with open("bandwidth-normalized.json", "w") as f:
    f.write(json.dumps(new_data, indent=4))

