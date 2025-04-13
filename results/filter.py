from pathlib import Path
import sys

from common import json_load, json_store


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py data.json 10")
        sys.exit(1)

    path = Path(sys.argv[1])
    x = float(sys.argv[2])  # e.g. 10 for top 10%

    data = json_load(path)

    # Convert to tuples of (name, entry, area)
    sized = []
    for name, vals in data.items():
        rows = int(vals["num_of_rows"])
        cols = int(vals["num_of_cols"])
        area = rows * cols
        sized.append((name, vals, area))

    # Sort by area descending
    sized.sort(key=lambda t: t[2], reverse=True)

    # Take top x%
    cutoff = max(1, int(len(sized) * x / 100)) # in case x is too small
    top = sized[:cutoff]

    # Return as dict
    filtered = {}
    for name, _, _ in top:
        filtered[name] = data[name]

    json_store(path.parent / (path.stem + f"_{int(x)}.json"), filtered)


if __name__ == "__main__":
    main()
