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

#ifndef NUM_OF_STREAMS
#define NUM_OF_STREAMS 1
#endif

#define STREAM_OFFSET_IN_CL 5

static char* addr[NUM_OF_STREAMS];
static int64_t cl_acc_times[NUM_OF_STREAMS][PAGE_SIZE_CL] = {0};


int main() {

    // Measure the overhead of measuring time
    uint32_t ignore;
    uint64_t before, after;
    double overhead_f = 0;
    for (int i = 1; i <= 100; i++) {
        _mm_mfence();
        before = __rdtsc();
        _mm_lfence();
        after = __rdtscp(&ignore);
        _mm_mfence();
        overhead_f += (1.0/i) * (after - before - overhead_f);
    }
    // An overestimate of the overhead
    const uint64_t overhead = (uint64_t)overhead_f;


    // Assuming you have already done something like:
    // echo 1 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
    char* const page_addr = mmap(NULL,
                                 PAGE_SIZE,
                                 PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE,
                                 -1, 0);
    if (page_addr == MAP_FAILED) {
        perror("mmap failed");
        exit(EXIT_FAILURE);
    }

    for (int s = 0; s < NUM_OF_STREAMS; s++) {
        addr[s] = page_addr + s * STREAM_OFFSET_IN_CL * CL;
    }

    // Touch every cache line in the page:
    // Even though we used MAP_POPULATE, let's make sure that everything is physically mapped
    for (volatile char *a = page_addr; a < page_addr + PAGE_SIZE; a += CL) {
        *a = 0;
    }

    // Flush the caches
    for (const char *a = page_addr; a < page_addr + PAGE_SIZE; a += CL) {
        _mm_clflush(a);
    }

    for (int i = 0; i < PAGE_SIZE_CL; i++) {
        #pragma GCC unroll NUM_OF_STREAMS
        for (int s = 0; s < NUM_OF_STREAMS; s++) {
            volatile char *ptr = addr[s] + i * CL;

            // Inspired by:
            // https://sites.utexas.edu/jdm4372/2018/07/23/comments-on-timing-short-code-sections-on-intel-processors/
            _mm_mfence();
            before = __rdtsc();
            _mm_lfence();
            (void)*ptr;
            after = __rdtscp(&ignore);
            _mm_mfence();

            cl_acc_times[s][i] = after - before - overhead;

            // The overhead is an overestimate so it may "hide" L1 hit times
            cl_acc_times[s][i] = cl_acc_times[s][i] > 0 ? cl_acc_times[s][i] : 1;
        }

        // Wait before issuing the next load to make sure that prefetches are placed in the cache
        uint64_t start_delay = __rdtsc();
        while (__rdtsc() - start_delay < 1000);
    }

    for (int s = 0; s < NUM_OF_STREAMS; s++) {
        munmap(addr[s], PAGE_SIZE);
        printf("STREAM %d:\n", s);
        for (int i = 0; i < PAGE_SIZE_CL; i++)
            printf("%"PRIu64"\n", cl_acc_times[s][i]);
    }

    return 0;
}
