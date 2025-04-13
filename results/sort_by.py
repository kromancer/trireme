import sys
import json
from common import json_load


if __name__ == '__main__':
    data = json_load(sys.argv[1])
    sort_by = sys.argv[2]
    sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: item[1].get(sort_by), reverse=True)}

    with open(sys.argv[1], "w") as f:
        f.write(json.dumps(sorted_data, indent=4))
