import argparse
from common import json_load, json_load_and_backup, json_store
from pathlib import Path

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-s", "--source", type=str, required=True)
    argparser.add_argument("-d", "--dest", type=str, required=True)
    argparser.add_argument("-f", "--field", type=str, required=True)
    argparser.add_argument("-r", "--rename", type=str, required=True)
    args = argparser.parse_args()

    src = json_load(args.source)

    dest_file = Path(args.dest)
    if dest_file.exists():
        dest = json_load_and_backup(args.dest)
    else:
        dest = {}

    for k, v in src.items():
        dest.setdefault(k, {})[args.rename] = v[args.field]

    json_store(args.dest, dest)



