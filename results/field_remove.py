import sys
from common import json_load_and_backup, json_store


if __name__ == "__main__":
    data = json_load_and_backup(sys.argv[1])
    field = sys.argv[2]

    for mtx, v in data.items():
        del v[field]

    json_store(sys.argv[1], data)
