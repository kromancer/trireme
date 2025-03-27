from argparse import ArgumentParser, Namespace
from contextlib import ExitStack
import mmap
import numpy as np
import os
import subprocess

from input_manager import InputManager


class RAMDisk:
    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("-1gb", "--use-1gb-pages-for-dense-arg", action="store_true",
                            help="Use 1GB pages for the vector to reduce TLB misses when prefetching")

    def __init__(self, args: Namespace, in_man: InputManager, dense_arg_buff: np.ndarray, *buffers: np.ndarray):
        self.ramdisks = []
        if args.use_1gb_pages_for_dense_arg:
            self.ramdisks.append(_RAMDisk("1G", in_man, dense_arg_buff))
            self.ramdisks.append(_RAMDisk("2M", in_man, *buffers))
        else:
            self.ramdisks.append(_RAMDisk("2M", in_man, dense_arg_buff, *buffers))

    def reset_res_buff(self):
        self.buffers[-1].fill(0)

    def __enter__(self):
        self._stack = ExitStack()
        try:
            for disk in self.ramdisks:
                self._stack.enter_context(disk)
            self.buffer_paths = [path for ramdisk in self.ramdisks for path in ramdisk.buffer_paths]
            self.buffers = [buf for ramdisk in self.ramdisks for buf in ramdisk.buffers]
            return self
        except Exception:
            self._stack.close()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self._stack.close()


class _RAMDisk:
    def __init__(self, page_size: str, in_man: InputManager, *buffers: np.ndarray):
        assert page_size in ["2M", "1G"], "Unsupported page size"
        self.page_size = page_size
        self.page_size_bytes = 2 * 2 ** 20 if self.page_size == "2M" else 2 ** 30
        self.page_size_in_kb = self.page_size_bytes // 1024
        self.buffers = buffers
        self.mount_point = f"/tmp/huge-{os.getpid()}-" + self.page_size
        self.buffer_sizes_in_pages = []
        self.mmaped = []
        self.buffer_paths = []
        self._calculate_total_pages()
        self.extra_pages = 0
        self.in_man = in_man

    def _calculate_total_pages(self):
        for buff in self.buffers:
            if not isinstance(buff, np.ndarray):
                raise ValueError("All provided buffers must be of type numpy.ndarray.")

            pages_needed = (buff.nbytes + self.page_size_bytes - 1) // self.page_size_bytes
            self.buffer_sizes_in_pages.append(pages_needed)

    def _reserve_hugepages(self):
        total_pages = sum(self.buffer_sizes_in_pages)
        kernel_cfg = f"/sys/kernel/mm/hugepages/hugepages-{self.page_size_in_kb}kB/nr_hugepages"

        try:
            # Read the current number of reserved huge pages
            with open(kernel_cfg, "r") as f:
                current_pages = int(f.read().strip())

            # Calculate how many extra pages are needed
            self.extra_pages = total_pages - current_pages

            if self.extra_pages > 0:
                with open(kernel_cfg, "w") as f:
                    f.write(str(total_pages))
                print(f"Reserved {self.extra_pages} extra huge pages of size {self.page_size}.")

        except PermissionError:
            raise RuntimeError(f"You need sudo/root privileges to modify {kernel_cfg}.")

    def _mount(self):
        if not os.path.exists(self.mount_point):
            os.makedirs(self.mount_point)
        try:
            subprocess.run(
                ["mount", "-t", "hugetlbfs", "-o", f"pagesize={self.page_size}", "none", self.mount_point],
                check=True,
            )
            print(f"Mounted hugetlbfs at {self.mount_point}.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to mount hugetlbfs: {e}")

    def _release_hugepages(self):
        if self.extra_pages == 0:
            return

        hugepages_path = f"/sys/kernel/mm/hugepages/hugepages-{self.page_size_in_kb}kB/nr_hugepages"
        try:
            # Read the current number of huge pages
            with open(hugepages_path, "r") as f:
                current_pages = int(f.read().strip())

            # Calculate new number of pages after releasing only the extras
            new_pages = max(0, current_pages - self.extra_pages)

            # Update sysfs only if there is a change
            if new_pages < current_pages:
                with open(hugepages_path, "w") as f:
                    f.write(str(new_pages))
                print(f"Released {self.extra_pages} huge pages. Remaining: {new_pages}.")
            else:
                print("No huge pages to release.")

        except PermissionError:
            print(f"Warning: You need sudo/root privileges to modify {hugepages_path}.")

    def _unmount(self):
        try:
            subprocess.run(["umount", self.mount_point], check=True)
            print(f"Unmounted hugetlbfs from {self.mount_point}.")
        except subprocess.CalledProcessError:
            print(f"Warning: Could not unmount {self.mount_point}. You may need to do this manually.")

    def _move_buffers(self):
        new_buffers = []
        for i, buf in enumerate(self.buffers):
            mmap_path = os.path.join(self.mount_point, f"buffer_{i}")
            self.buffer_paths.append(mmap_path)

            aligned_size = self.buffer_sizes_in_pages[i] * self.page_size_bytes
            with open(mmap_path, "wb") as f:
                f.truncate(aligned_size)  # Resize the file to buffer size

            # Memory-map the file
            with open(mmap_path, "r+b") as f:
                mmapped = mmap.mmap(f.fileno(), aligned_size, access=mmap.ACCESS_WRITE)
                self.mmaped.append(mmapped)

                # Create a new ndarray backed by the memory-map
                new_buf = np.ndarray(shape=buf.shape, dtype=buf.dtype, buffer=mmapped)
                new_buf[:] = buf  # Copy data to the new buffer
                new_buffers.append(new_buf)

        self.buffers = tuple(new_buffers)
        print(f"Buffers have been successfully moved to {self.mount_point}.")

    def __enter__(self):
        # no cleanup required here if this action excepts
        self._reserve_hugepages()

        # if this excepts, simply release the number of reserved pages
        try:
            self._mount()
        except Exception as e:
            self._release_hugepages()
            raise RuntimeError(f"Failed to mount HugeTLBFS: {e}")

        # if this excepts, release and unmount
        try:
            self._move_buffers()
            return self
        except Exception as e:
            self._release_hugepages()
            self._unmount()
            raise RuntimeError(f"Failed to move buffers: {e}")

    def __exit__(self, exc_type, exc_value, traceback):
        for mmapped in self.mmaped:
            mmapped.close()
        self._release_hugepages()
        self._unmount()
