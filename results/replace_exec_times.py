import argparse
from common import json_load_and_backup, json_load
import json
import sys


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-s", "--source", type=str, required=True)
    argparser.add_argument("-d", "--dest", type=str, required=True)
    args = argparser.parse_args()

    dest_data = json_load_and_backup(args.dest)
    src_data = json_load(args.source)

    for mtx, v in dest_data.items():
        for field in ["exec_times_ns", "mean_ms", "std_dev", "cv%"]:
            v[field] = src_data[mtx][field]

        for field in ["args", "time", "git-hash"]:
            v[field] = [v[field], src_data[mtx][field]]

    with open(sys.argv[1], "w") as f:
        f.write(json.dumps(dest_data, indent=4))


