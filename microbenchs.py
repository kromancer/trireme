import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from subprocess import run

from common import build_with_cmake
from hwpref_controller import HwprefController


def plot_time_series(file_path, output_pdf="mem_latencies_per_cl.pdf"):

    # Define saturation function
    def saturate(value, max_value=200):
        return min(value, max_value)  # Clamp value to max_value

    # Read data from file
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Parse the data into separate regions
    streams = {}
    current_stream = None

    for line in lines:
        line = line.strip()
        if line.startswith("STREAM"):
            # Extract stream number
            current_stream = int(line.split()[1][:-1])  # Extract number from "STREAM X:"
            streams[current_stream] = []
        elif current_stream is not None and line.isdigit():
            streams[current_stream].append(int(line))

    # Plot the time series
    plt.figure(figsize=(10, 5))

    for stream_id, data in streams.items():
        saturated_data = [saturate(x) for x in data[0:50]]
        plt.plot(saturated_data, marker="o", linestyle="-", markersize=2, label=f"STREAM {stream_id}")

    plt.xlabel("CL#")
    plt.ylabel("Cycles")
    plt.title("Mem Latency per CL access over the first CLs of a 2MB page")
    plt.grid(True)

    # Save as PDF
    plt.savefig(output_pdf)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    HwprefController.add_args(parser)
    parser.add_argument("-s", "--num_of_streams", type=int, default=1, help="Number of Streams")
    parser.add_argument("-o", "--variant", choices=["one_page_per_stream", "one_page_many_streams"],
                        help="Choose an analysis type")
    args = parser.parse_args()

    src_path = Path(__file__).parent.resolve() / "microbenchs"
    exe = build_with_cmake([f"-DNUM_OF_STREAMS={args.num_of_streams}"], args.variant, src_path)

    with HwprefController(args) as hpc:
        result = run(exe, check=True, text=True, capture_output=True)

    # Save stdout to a text file
    out = "mem_latencies_per_cl.txt"
    with open(out, "w") as f:
        f.write(result.stdout)

    plot_time_series(out)
