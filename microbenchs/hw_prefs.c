#include <emmintrin.h>
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

    uint64_t cl_acc_times[PAGE_SIZE_CL] = {0};

    // Assuming you have run something like:
    // echo 1 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
    char* const addr = mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE,
                -1, 0);

    if (addr == MAP_FAILED) {
        perror("mmap failed");
        exit(EXIT_FAILURE);
    }

    // Touch every cache line in the page:
    // Even though we used MAP_POPULATE, let's make sure that everything is physically mapped
    for (volatile char *a = addr; a < addr + PAGE_SIZE; a += CL) {
        *a = 0;
    }

    // Flush the caches
    for (const char *a = addr; a < addr + PAGE_SIZE; a += CL) {
        _mm_clflush(a);
    }
    _mm_mfence();

    uint32_t ignore;
    const volatile char* a = addr;
    for (int i = 0; i < PAGE_SIZE_CL; i++) {

        // Inspired by:
        // https://sites.utexas.edu/jdm4372/2018/07/23/comments-on-timing-short-code-sections-on-intel-processors/
        uint64_t before = __rdtsc();
        _mm_lfence();
        (void)*a;
        uint64_t after = __rdtscp(&ignore);

        _mm_mfence();
        cl_acc_times[i] = after - before;
        a += CL;
        _mm_mfence();
    }

    for (int i = 0; i < PAGE_SIZE_CL; i++)
        printf("%"PRIu64"\n", cl_acc_times[i]);

    munmap(addr, PAGE_SIZE);
    return 0;
}
