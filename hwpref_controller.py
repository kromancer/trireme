from argparse import ArgumentParser
import ctypes
from pathlib import Path

from common import build_with_cmake


class HwprefController:

    @staticmethod
    def add_args(arg_parser: ArgumentParser):
        arg_parser.add_argument('--disable-l1-ipp', action='store_true',
                                help='Disable the L1 Instruction Point Prefetcher')
        arg_parser.add_argument('--disable-l1-npp', action='store_true',
                                help='Disable the L1 Next Page Prefetcher')
        arg_parser.add_argument('--disable-l2-stream', action='store_true',
                                help='Disable the L2 Stream Prefetcher')
        arg_parser.add_argument('--disable-l2-amp', action='store_true',
                                help='Disable the L2 Adaptive Multi-Path Prefetcher')
        arg_parser.add_argument('--disable-llc-stream', action='store_true',
                                help='Disable the LLC Stream Prefetcher')

    @staticmethod
    def get_cmake_args(args):
        return [f"-DDISABLE_HW_PREF_L1_IPP={1 if args.disable_l1_ipp else 0}",
                f"-DDISABLE_HW_PREF_L1_NPP={1 if args.disable_l1_npp else 0}",
                f"-DDISABLE_HW_PREF_L2_STREAM={1 if args.disable_l2_stream else 0}",
                f"-DDISABLE_HW_PREF_L2_AMP={1 if args.disable_l2_amp else 0}",
                f"-DDISABLE_HW_PREF_LLC_STREAM={1 if args.disable_llc_stream else 0}"]

    def __init__(self, args):

        self.skip = any([args.disable_l1_ipp, args.disable_l1_npp, args.disable_l2_stream, args.disable_l2_amp,
                         args.disable_llc_stream])
        if self.skip:
            return

        cmake_args = HwprefController.get_cmake_args(args)

        src_path = Path(__file__).parent.resolve() / "hwpref_ctrl"
        lib_path = build_with_cmake(cmake_args=cmake_args, target="hwpref_ctrl_wrapper", src_path=src_path, is_lib=True)
        lib = ctypes.CDLL(str(lib_path))

        # Set up the function prototypes
        self.init_hw_pref_control = lib.init_hw_pref_control
        self.init_hw_pref_control.restype = ctypes.c_int

        self.deinit_hw_pref_control = lib.deinit_hw_pref_control
        self.deinit_hw_pref_control.restype = None

    def __enter__(self):
        if self.skip:
            return
        assert self.init_hw_pref_control() >= 0

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.skip:
            return
        self.deinit_hw_pref_control()
