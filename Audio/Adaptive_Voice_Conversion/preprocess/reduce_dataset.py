import json
import sys

pkl_path = sys.argv[1]
output_path = sys.argv[2]
segment_size = int(sys.argv[3])

with open(pkl_path, "r") as f:
    data = json.load(f)

reduced_data = {key: val for key, val in data.items() if val['shape'][0] > segment_size}

with open(output_path, "w") as f:
    json.dump(reduced_data, f)
