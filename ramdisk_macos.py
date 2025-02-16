from argparse import ArgumentParser, Namespace
import mmap
import numpy as np
from pathlib import Path
from subprocess import run

from input_manager import InputManager


class RAMDisk:
    @staticmethod
    def add_args(parser: ArgumentParser):
        pass

    def __init__(self, args: Namespace, in_man: InputManager, *buffers: np.ndarray):
        self.ramdisk_name = "ramdisk"
        self.mount_point = None
        self.ramdisk_device = None
        self.buffers = buffers
        self.buffer_paths = []
        self.mmaped = []
        self.in_man = in_man

    def _mount(self):
        total_bytes = sum(buf.nbytes for buf in self.buffers)
        overhead_factor = 1.2
        adjusted_total_bytes = int(overhead_factor * total_bytes)
        sectors = (adjusted_total_bytes + 511) // 512

        # HFS+ expects a minimum valid size, let's say 8MB
        if sectors < 16384:
            sectors = 16384

        # Create the RAM disk device
        attach_cmd = ["hdiutil", "attach", "-nomount", f"ram://{sectors}"]
        result = run(attach_cmd, capture_output=True, text=True, check=True)
        self.ramdisk_device = result.stdout.strip()

        # Format and mount as HFS+ (ends up in /Volumes/ramdisk)
        erase_cmd = ["diskutil", "erasevolume", "HFS+", self.ramdisk_name, self.ramdisk_device]
        run(erase_cmd, check=True)
        self.mount_point = f"/Volumes/{self.ramdisk_name}"
        print(f"Mounted a RAM disk at {self.mount_point}")

    def _move_buffers(self):
        assert self.mount_point is not None, "Ramdisk is not mounted!"

        new_buffers = []
        for i, buf in enumerate(self.buffers):
            # Path for the hugepage-backed storage
            mmap_path = Path(self.mount_point, f"buffer_{i}")
            self.buffer_paths.append(mmap_path)

            with open(mmap_path, "wb") as f:
                f.truncate(buf.nbytes)

            # Memory-map the file
            with open(mmap_path, "r+b") as f:
                mmapped = mmap.mmap(f.fileno(), buf.nbytes, access=mmap.ACCESS_WRITE)
                self.mmaped.append(mmapped)

                # Create a new ndarray backed by the memory-map
                new_buf = np.ndarray(shape=buf.shape, dtype=buf.dtype, buffer=mmapped)
                new_buf[:] = buf
                new_buffers.append(new_buf)

        self.buffers = tuple(new_buffers)
        print(f"Buffers have been successfully moved to {self.mount_point}.")

    def _eject(self):
        assert self.mount_point is not None, "Ramdisk is not mounted!"
        run(["diskutil", "eject", self.ramdisk_device])

    def reset_res_buff(self):
        self.buffers[-1].fill(0)

    def __enter__(self):
        self._mount()
        try:
            self._move_buffers()
        except Exception as e:
            self._eject()
            raise RuntimeError(f"Failed to move buffers: {e}")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for mmapped in self.mmaped:
            mmapped.close()
        try:
            self._eject()
        except Exception as e:
            print(f"Warning: Could not eject {self.mount_point}. You may need to do this manually.")

