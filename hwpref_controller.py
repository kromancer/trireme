from argparse import ArgumentParser, ArgumentTypeError
import ctypes
from pathlib import Path

from common import build_with_cmake


class HwprefController:

    @staticmethod
    def add_args(arg_parser: ArgumentParser):
        arg_parser.add_argument('--disable-l1-nlp', action='store_true',
                                help='Disable the L1 Next Line Prefetcher')
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

        def in_range_validator(min_val, max_val):
            def validator(val):
                i = int(val)
                if i < min_val or i > max_val:
                    raise ArgumentTypeError(f"Value must be between {min_val} and {max_val}")
                return i
            return validator

        arg_parser.add_argument('--l2-stream-dd', type=in_range_validator(-1, 255),
                                metavar='[-1, 255]', default=-1,
                                help="Set the L2 Stream's Demand Density Threshold")
        arg_parser.add_argument('--l2-stream-dd-ovr', type=in_range_validator(-1, 16),
                                metavar='[-1, 16]', default=-1,
                                help="Set the L2 Stream's Demand Density Threshold Override")
        arg_parser.add_argument('--l2-stream-xq-thres', type=in_range_validator(-1, 32),
                                metavar='[-1, 32]', default=-1,
                                help="Set the L2 Stream's XQ Threshold")

    @staticmethod
    def get_cmake_args(args):
        return [f"-DDISABLE_HW_PREF_L1_NLP={1 if args.disable_l1_nlp else 0}",
                f"-DDISABLE_HW_PREF_L1_IPP={1 if args.disable_l1_ipp else 0}",
                f"-DDISABLE_HW_PREF_L1_NPP={1 if args.disable_l1_npp else 0}",
                f"-DDISABLE_HW_PREF_L2_STREAM={1 if args.disable_l2_stream else 0}",
                f"-DDISABLE_HW_PREF_L2_AMP={1 if args.disable_l2_amp else 0}",
                f"-DDISABLE_HW_PREF_LLC_STREAM={1 if args.disable_llc_stream else 0}",
                f"-DSET_L2_STREAM_DD={args.l2_stream_dd}",
                f"-DSET_L2_STREAM_DD_OVR={args.l2_stream_dd_ovr}",
                f"-DSET_L2_STREAM_XQ_THRES={args.l2_stream_xq_thres}"]

    def __init__(self, args):

        self.skip = (not any([args.disable_l1_ipp, args.disable_l1_npp, args.disable_l1_nlp,
                             args.disable_l2_stream, args.disable_l2_amp, args.disable_llc_stream])
                     and args.l2_stream_dd == -1 and args.l2_stream_dd_ovr == -1 and args.l2_stream_xq_thres == -1)
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
            return self
        if self.init_hw_pref_control() < 0:
            print("Failed to initiliaze HW Pref Controls")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.skip:
            return False
        self.deinit_hw_pref_control()
