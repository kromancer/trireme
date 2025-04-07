import json
from pathlib import Path
from suite_sparse import SuiteSparse



def old():
    with open("consolidated-no-opt.json", "r") as f:
        data = json.load(f)

    ss = SuiteSparse(Path("."))
    new_data = {}
    for mtx, v in data.items():
        num_of_cols = ss.get_meta(mtx, "num_of_cols")
        elem_size = 8 if ss.get_meta(mtx, "is_binary") != 1 else 1
        new_data[mtx] = int(num_of_cols) * elem_size

    with open("vector-sizes.json", "w") as f:
        f.write(json.dumps(new_data, indent=4))
