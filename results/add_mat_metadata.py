import json
from pathlib import Path
import sys
from suite_sparse import SuiteSparse

from common import print_size

if __name__ == "__main__":
    rep = Path(sys.argv[1])
    assert Path(rep).exists(), f"{rep} does not exist"

    with open(rep, "r") as f, open(rep.parent / ("bak-" + rep.name), "w") as bak:
        data = json.load(f)
        bak.write(json.dumps(data, indent=4))

    ss = SuiteSparse(Path("."))
    for mtx, v in data.items():

        for meta in ss.column_headers:
            v[meta] = str(ss.get_meta(mtx, meta))

        if ss.get_meta(mtx, "is_binary") == 1:
            elem_size = 1
            elems_per_cl = 8
        else:
            elem_size = 8
            elems_per_cl = 64

        v["vec_size"] = print_size(int(v["num_of_cols"]) * elem_size)
        v["mat_size"] = print_size(int(v["num_of_cols"]) * elems_per_cl * elem_size)
        v["density"] = int(v["num_of_entries"]) / (int(v["num_of_cols"]) * int(v["num_of_rows"]))

    with open(rep, "w") as f:
        f.write(json.dumps(data, indent=4))
