#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "spmv.h"

struct shm_info {
    void *ptr;
    struct stat sb;
};

struct shm_info open_and_map_shm(const char *shm, bool is_read_only) {

    // Open the shared memory object
    int shm_fd = shm_open(shm, is_read_only ? O_RDONLY : O_RDWR, 0666);
    if (shm_fd == -1) {
        perror("shm_open");
        exit(EXIT_FAILURE);
    }

    // Map the shared memory object
    struct stat sb;
    if (fstat(shm_fd, &sb) == -1) {
        perror("fstat");
        exit(EXIT_FAILURE);
    }

    void *ptr = mmap(0, sb.st_size, is_read_only ? PROT_READ : PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        exit(EXIT_FAILURE);
    }

    // Close the file descriptor after mapping the memory
    close(shm_fd);

    // Create a ShmInfo struct to hold both values and return it
    struct shm_info result = {ptr, sb};
    return result;
}

uint64_t get_uint64(const char *arg) {

    // Convert the command-line argument (string) to an integer using strtol
    char *endptr;
    errno = 0; // To distinguish success/failure after call
    uint64_t num = strtol(arg, &endptr, 10);

    // Check for conversion errors and out-of-range values
    if ( (errno == ERANGE && num == UINT64_MAX) || (errno != 0 && num == 0) ) {
        perror("strtol");
        exit(EXIT_FAILURE);
    }

    // Check if there are any non-numeric characters in the input
    if (*endptr != '\0') {
        printf("Invalid integer: %s\n", arg);
        exit(EXIT_FAILURE);
    }

    return num;
}

int main(int argc, char **argv) {

    if (argc < 7) {
        fprintf(stderr, "Usage: %s <num_of_rows> <vec> <mat.data> <mat.indices> <mat.indptr> <res>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    uint64_t num_of_rows = get_uint64(argv[1]);

    struct shm_info vec = open_and_map_shm(argv[2], true);
    struct shm_info mat_data = open_and_map_shm(argv[3], true);
    struct shm_info mat_indptr = open_and_map_shm(argv[4], true);
    struct shm_info mat_indices = open_and_map_shm(argv[5], true);
    struct shm_info res = open_and_map_shm(argv[6], false);

    double exec_time = compute(
            num_of_rows,
            (const double *)vec.ptr,
            (const double *)mat_data.ptr,
            (const int64_t *)mat_indptr.ptr,
            (const int64_t *)mat_indices.ptr,
            (double *)res.ptr);

    // Clean up
    munmap(vec.ptr, vec.sb.st_size);
    munmap(mat_data.ptr, mat_data.sb.st_size);
    munmap(mat_indices.ptr, mat_indices.sb.st_size);
    munmap(mat_indptr.ptr, mat_indptr.sb.st_size);
    munmap(res.ptr, res.sb.st_size);

    printf("exec time: %fs\n", exec_time);
    return 0;
}
