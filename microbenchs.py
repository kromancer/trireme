import argparse
from pathlib import Path
from subprocess import run

from common import build_with_cmake
from hwpref_controller import HwprefController

if __name__ == "__main__":
    src_path = Path(__file__).parent.resolve() / "microbenchs"
    exe = build_with_cmake([], "hw_prefs", src_path)

    parser = argparse.ArgumentParser()
    HwprefController.add_args(parser)
    args = parser.parse_args()

    with HwprefController(args) as hpc:
        run(exe, check=True)
