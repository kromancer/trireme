import argparse
import json
from os import chdir, close, dup, dup2, fsync, makedirs
from pathlib import Path
from shutil import rmtree, which
import sys
from tempfile import TemporaryFile
from typing import Any, Callable, List, Tuple


def read_config(file: str, key: str) -> List[str]:
    script_dir = Path(__file__).parent.resolve()
    cfg_file = script_dir / file

    cfg = []
    try:
        with open(cfg_file, "r") as f:
            print(f"Reading config {cfg_file}")
            cfg = json.load(f)[key]
    except FileNotFoundError:
        print(f"No {file} in {script_dir}")
    except json.decoder.JSONDecodeError as e:
        print(f"{cfg_file} could not be decoded {e}")
    except KeyError:
        print(f"{cfg_file} does not have a field for {key}")
    finally:
        return cfg


def get_spmv_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("-r", "--rows", type=int, default=1024,
                               help="Number of rows (default=1024)")
    parser.add_argument("-c", "--cols", type=int, default=1024,
                               help="Number of columns (default=1024)")
    parser.add_argument("-pd", "--prefetch-distance", type=int, default=16,
                               help="Prefetch distance")
    parser.add_argument("-l", "--locality-hint", type=int, choices=[0, 1, 2, 3], default=3,
                        help="Temporal locality hint for prefetch instructions, "
                             "3 for maximum temporal locality, 0 for no temporal locality. "
                             "On x86, value 3 will produce PREFETCHT0, while value 0 will produce PREFETCHNTA")
    parser.add_argument("-d", "--density", type=float, default=0.05,
                        help="Density of sparse matrix (default=0.05)")

    return parser


def make_work_dir_and_cd_to_it(file_path: str):
    build_path = Path(f"./workdir-{Path(file_path).name}")

    if build_path.exists():
        rmtree(build_path)
    makedirs(build_path)

    chdir(build_path)


def is_in_path(exe: str) -> bool:

    exe_path = which(exe)

    if exe_path is None:
        print(f"{exe} not in PATH")
        return False

    return True


def run_func_and_capture_stdout(func: Callable, *args, **kwargs) -> Tuple[str, Any]:
    # Backup the original stdout file descriptor
    stdout_fd = sys.stdout.fileno()
    original_stdout_fd = dup(stdout_fd)

    # Create a temporary file to capture the output
    with TemporaryFile(mode='w+b') as tmpfile:
        # Redirect stdout to the temporary file
        dup2(tmpfile.fileno(), stdout_fd)

        try:
            # Call the function
            res = func(*args, **kwargs)

            # Flush any buffered output
            fsync(stdout_fd)

            # Go back to the start of the temporary file to read its contents
            tmpfile.seek(0)
            captured = tmpfile.read().decode()
        finally:
            # Restore the original stdout file descriptor
            dup2(original_stdout_fd, stdout_fd)
            close(original_stdout_fd)

    return captured, res


def print_size(size):
    KB = 1024
    MB = KB ** 2
    GB = KB ** 3

    if size >= GB:
        return f"{size / GB:.2f} GB"
    elif size >= MB:
        return f"{size / MB:.2f} MB"
    elif size >= KB:
        return f"{size / KB:.2f} KB"
    else:
        return f"{size} Bytes"
