import os
import json
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def compute_bandwidths(directory, output_file):
    bandwidth_dict = {}

    for filename in os.listdir(directory):
        if filename.endswith(".npz"):
            file_path = os.path.join(directory, filename)
            matrix_name = os.path.splitext(filename)[0]  # Remove .npz extension

            try:
                matrix = sp.load_npz(file_path)
                bandwidth = spla.spbandwidth(matrix)
                bandwidth_dict[matrix_name] = bandwidth
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    with open(output_file, "w") as f:
        json.dump(bandwidth_dict, f, indent=4)


if __name__ == "__main__":
    directory = "/Users/ioanniss/trireme-inputs"
    output_file = "bandwidth_results.json"
    compute_bandwidths(directory, output_file)
    print(f"Results saved in {output_file}")
