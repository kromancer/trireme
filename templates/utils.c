#include "utils.h"

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>


uint64_t get_uint64(const char *arg) {

    // Convert the command-line argument (string) to an integer using strtol
    char *endptr;
    errno = 0; // To distinguish success/failure after call
    uint64_t num = strtoull(arg, &endptr, 10);

    // Check for conversion errors and out-of-range values
    if ( (errno == ERANGE && num == UINT64_MAX) || (errno != 0 && num == 0) ) {
        perror("strtoull");
        exit(EXIT_FAILURE);
    }

    // Check if there are any non-numeric characters in the input
    if (*endptr != '\0') {
        printf("Invalid integer: %s\n", arg);
        exit(EXIT_FAILURE);
    }

    return num;
}

struct shm_info open_and_map_shm(const char *shm, bool is_read_only) {
    int shm_fd = -1;

    // Open the shared memory object
    shm_fd = open(shm, O_RDWR);
    if (shm_fd == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    // Map the shared memory object
    struct stat sb;
    if (fstat(shm_fd, &sb) == -1) {
        perror("fstat");
        close(shm_fd);
        exit(EXIT_FAILURE);
    }

    void *ptr = mmap(0, sb.st_size, is_read_only ? PROT_READ : PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        close(shm_fd);
        exit(EXIT_FAILURE);
    }

    // Close the file descriptor after mapping the memory
    close(shm_fd);

    // Create a ShmInfo struct to hold both values and return it
    struct shm_info result = {ptr, sb};
    return result;
}
