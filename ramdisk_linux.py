from argparse import ArgumentParser, Namespace
import gc
import mmap
import numpy as np
import os
import subprocess


class RAMDisk:
    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("-1gb", "--use-1gb-pages-for-vec", action="store_true",
                            help="Use 1GB pages for the vector to reduce TLB misses when prefetching")

    def __init__(self, args: Namespace, vec_buff: np.ndarray, *buffers: np.ndarray):
        self.ramdisks = []
        if args.use_1gb_pages_for_vec:
            self.ramdisks.append(_RAMDisk("1G", vec_buff))
            self.ramdisks.append(_RAMDisk("2M", *buffers))
        else:
            self.ramdisks.append(_RAMDisk("2M", vec_buff, *buffers))

    def __enter__(self):
        self.entered_ramdisks = [ramdisk.__enter__() for ramdisk in self.ramdisks]
        self.buffer_paths = [path for ramdisk in self.entered_ramdisks for path in ramdisk.buffer_paths]
        self.buffers = [buf for ramdisk in self.entered_ramdisks for buf in ramdisk.buffers]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Exit in reverse order
        for manager in reversed(self.ramdisks):
            manager.__exit__(exc_type, exc_value, traceback)


class _RAMDisk:
    def __init__(self, page_size: str, *buffers: np.ndarray):
        assert page_size in ["2M", "1G"], "Unsupported page size"
        self.page_size = page_size
        self.page_size_bytes = 2 * 2 ** 20 if self.page_size == "2M" else 2 ** 30
        self.page_size_in_kb = self.page_size_bytes // 1024
        self.buffers = buffers
        self.mount_point = "/tmp/huge-" + self.page_size
        self.buffer_sizes_in_pages = []
        self.mmaped = []
        self.buffer_paths = []
        self._calculate_total_pages()

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
            with open(kernel_cfg, "w") as f:
                f.write(str(total_pages))
            print(f"Reserved {total_pages} huge pages of size {self.page_size}.")
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

    def _unmount(self):
        try:
            subprocess.run(["umount", self.mount_point], check=True)
            print(f"Unmounted hugetlbfs from {self.mount_point}.")
        except subprocess.CalledProcessError:
            print(f"Warning: Could not unmount {self.mount_point}. You may need to do this manually.")

        hugepages_path = f"/sys/kernel/mm/hugepages/hugepages-{self.page_size_in_kb}kB/nr_hugepages"
        try:
            with open(hugepages_path, "w") as f:
                f.write("0")
            print(f"Released all reserved huge pages.")
        except PermissionError:
            print("Warning: You need root privileges to release huge pages.")
        except Exception as e:
            print(f"Warning: Could not release huge pages: {e}")

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

                # Deallocate the old buffer
                del buf
                gc.collect()

        self.buffers = tuple(new_buffers)
        print(f"Buffers have been successfully moved to {self.mount_point}.")

    def __enter__(self):
        try:
            self._reserve_hugepages()
            self._mount()
            self._move_buffers()
        except Exception as e:
            raise RuntimeError(f"Failed to set up HugeTLBFS: {e}")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for mmapped in self.mmaped:
            mmapped.close()
        self._unmount()
