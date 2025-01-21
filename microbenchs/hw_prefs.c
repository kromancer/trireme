#define _GNU_SOURCE

#include <x86intrin.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define PAGE_SIZE (2 * 1024 * 1024)
#define CL 64
#define PAGE_SIZE_CL (PAGE_SIZE) / CL


int main() {

    uint64_t cl_acc_times[PAGE_SIZE_CL + 1] = {0};

    // Assuming you have run something like:
    // echo 1 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
    const char* const addr = mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE,
                -1, 0);

    if (addr == MAP_FAILED) {
        perror("mmap failed");
        exit(EXIT_FAILURE);
    }

    const volatile char *a = addr;
    for (; a < addr + PAGE_SIZE; a += CL) {
        _mm_clflush(addr);
    }

    a = addr;
    int i = 1;
    const char* const page_lim = addr + PAGE_SIZE;

    _mm_mfence();

    *cl_acc_times[0] = __rdtsc();
    #pragma GCC unroll 8
    for (; a < page_lim; a += CL, i++) {
        (void)*a;
        cl_acc_times[i] = __rdtsc();
    }

    for (int i = 1; i < 1 + PAGE_SIZE_CL; i++)
        printf("%"PRIu64"\n", cl_acc_times[i] - cl_acc_times[i - 1]);

    munmap(addr, PAGE_SIZE);
    return 0;
}
