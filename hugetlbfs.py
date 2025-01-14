import mmap
import numpy as np
import os
import platform
import subprocess


class HugeTLBFS:
    """
    A class to manage HugeTLBFS, providing a context manager for easier setup and cleanup.
    When not running on Linux, the class does nothing.
    """

    def __init__(self, pagesize, *buffers):
        """
        Initialize the HugeTLBFS instance.

        Parameters:
        - mount_point (str): Directory to mount the hugetlbfs.
        - pagesize (str): Huge page size, e.g., "2MB" or "1GB".
        - *buffers (np.ndarray): Arbitrary numpy arrays whose memory needs to be reserved.

        Requires sudo/root privileges.
        """
        self.mount_point = "/hugetlbfs"
        self.pagesize = pagesize.upper()
        self.buffers = buffers
        self.buffer_sizes_in_pages = []
        self.buffer_paths = []
        self.pagesize_kb = None  # Converted page size in kilobytes
        self.num_pages = 0  # Total pages to be allocated

        # Check if running on Linux
        self._is_linux = platform.system() == "Linux"

        if self._is_linux:
            # Validate and set the page size
            if self.pagesize == "2MB":
                self.pagesize_kb = 2048
                self.page_size_bytes = 2 * 1024 * 1024  # 2MB in bytes
            elif self.pagesize == "1GB":
                self.pagesize_kb = 1048576
                self.page_size_bytes = 1 * 1024 * 1024 * 1024  # 1GB in bytes
            else:
                raise ValueError("Invalid page size. Supported options are '2MB' or '1GB'.")

            # Calculate the total number of pages required
            self.num_pages = self._calculate_total_pages()

    def _calculate_total_pages(self):
        """
        Calculate the total number of pages required to accommodate all provided numpy arrays.

        Parameters:
        - page_size (int): The page size in bytes.
        - *arrays (np.ndarray): Arbitrary number of numpy arrays.

        Returns:
        - int: Total number of pages required.
        """
        total_pages = 0

        for array in self.buffers:
            if not isinstance(array, np.ndarray):
                raise ValueError("All provided buffers must be of type numpy.ndarray.")

            # Calculate size of the array in bytes
            buffer_size_in_bytes = array.nbytes

            # Calculate the number of pages required for this buffer (round up)
            pages_needed = (buffer_size_in_bytes + self.pagesize - 1) // self.pagesize
            self.buffer_sizes_in_pages.append(pages_needed)

        self.num_pages = sum(self.buffer_sizes_in_pages)

    def _reserve_hugepages(self):
        """
        Reserve the requested number of huge pages.
        """
        hugepages_path = f"/sys/kernel/mm/hugepages/hugepages-{self.pagesize_kb}kB/nr_hugepages"
        try:
            with open(hugepages_path, "w") as f:
                f.write(str(self.num_pages))
            print(f"Reserved {self.num_pages} huge pages of size {self.pagesize}.")
        except FileNotFoundError:
            raise RuntimeError(f"Your kernel does not support {self.pagesize} huge pages.")
        except PermissionError:
            raise RuntimeError("You need sudo/root privileges to modify huge pages.")

    def _mount_hugetlbfs(self):
        """
        Mount the hugetlbfs at the specified mount point.
        """
        # Create the mount point directory if it doesn't exist
        if not os.path.exists(self.mount_point):
            os.makedirs(self.mount_point)
        try:
            subprocess.run(
                ["mount", "-t", "hugetlbfs", "-o", f"pagesize={self.pagesize}", "none", self.mount_point],
                check=True,
            )
            print(f"Mounted hugetlbfs at {self.mount_point} with page size {self.pagesize}.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to mount hugetlbfs: {e}")

    def _unmount_hugetlbfs(self):
        """
        Unmount the hugetlbfs and release the reserved huge pages.
        """
        # Unmount the file system
        try:
            subprocess.run(["umount", self.mount_point], check=True)
            print(f"Unmounted hugetlbfs from {self.mount_point}.")
        except subprocess.CalledProcessError:
            print(f"Warning: Could not unmount {self.mount_point}. You may need to do this manually.")

        # Release the huge pages
        hugepages_path = f"/sys/kernel/mm/hugepages/hugepages-{self.pagesize_kb}kB/nr_hugepages"
        try:
            with open(hugepages_path, "w") as f:
                f.write("0")
            print(f"Released all reserved {self.pagesize} huge pages.")
        except PermissionError:
            print("Warning: You need root privileges to release huge pages.")
        except Exception as e:
            print(f"Warning: Could not release huge pages: {e}")

    def _move_to_hugetlbfs(self):
        """
        Move all buffers in self.buffers to new numpy arrays backed by hugetlbfs.

        Deallocates the memory of the original numpy arrays after the move.
        """
        if not self._is_linux:
            print("Not running on Linux. Cannot move buffers to hugetlbfs.")
            return

        new_buffers = []
        for i, buf in enumerate(self.buffers):
            # Path for the hugepage-backed storage
            mmap_path = os.path.join(self.mount_point, f"buffer_{i}")
            self.buffer_paths.append(mmap_path)

            # Create a memory-mapped file at the hugetlbfs mount point
            aligned_size = self.buffer_sizes_in_pages[i] * self.page_size_bytes
            with open(mmap_path, "wb") as f:
                f.truncate(self.buffer_sizes_in_pages[i] * self.page_size_bytes)  # Resize the file to buffer size

            # Memory-map the file
            with open(mmap_path, "r+b") as f:
                mmapped = mmap.mmap(f.fileno(), aligned_size, access=mmap.ACCESS_WRITE)

                # Create a new ndarray backed by the memory-map
                new_buf = np.ndarray(shape=buf.shape, dtype=buf.dtype, buffer=mmapped)
                new_buf[:] = buf  # Copy data to the new buffer
                new_buffers.append(new_buf)

                # Deallocate the old buffer
                del buf

        self.buffers = tuple(new_buffers)
        print("Buffers have been successfully moved to hugetlbfs.")

    def __enter__(self):
        """
        Enter the context: Reserve and mount hugetlbfs.
        """
        if self._is_linux:
            try:
                self._reserve_hugepages()
                self._mount_hugetlbfs()
                self._move_to_hugetlbfs()
            except Exception as e:
                raise RuntimeError(f"Failed to set up HugeTLBFS: {e}")
        else:
            print("Not running on Linux. HugeTLBFS will do nothing.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context: Unmount and release resources.
        """
        if self._is_linux:
            self._unmount_hugetlbfs()
        else:
            print("Nothing to clean up as HugeTLBFS was not set up.")
