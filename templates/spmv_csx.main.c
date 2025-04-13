#define _GNU_SOURCE
#include <sched.h>

#include "utils.h"

#include <assert.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <errno.h>

// #define CYCLES_INDEX 0
#define L3_HIT_INDEX 0

#define INST_INDEX 1

// #define ALL_LOADS_INDEX 2
#define L2_HIT_INDEX 2

#define LLC_MISSES_INDEX 3
#define NUM_EVENTS 4

long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu,
                            int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

// Configure perf events
void perf_configure_events(struct perf_event_attr *pea) {
    static const uint64_t e_cfg[][2] = {
        // [CYCLES_INDEX] = {0x00c0, 10},
        [L3_HIT_INDEX] = {0x04d1, 10},
        [INST_INDEX] = {0x00c2, 10},
        // [ALL_LOADS_INDEX] = {0x81d0, 10},
        [L2_HIT_INDEX] = {0x02d1, 10},
        [LLC_MISSES_INDEX] = {0x80d1, 10}

    };

    // Configure each event
    for (int i = 0; i < NUM_EVENTS; i++) {
        memset(&pea[i], 0, sizeof(struct perf_event_attr));
        pea[i].size = sizeof(struct perf_event_attr);
        pea[i].disabled = 1;
        pea[i].exclude_kernel = 1;
        pea[i].exclude_hv = 1;
        pea[i].exclude_idle = 1;
        pea[i].exclude_guest = 1;
        pea[i].config = e_cfg[i][0];
        pea[i].type = e_cfg[i][1];
    }
}

void perf_init(int event_fds[NUM_EVENTS]) {
    int cpu = sched_getcpu();

    static struct perf_event_attr event_attrs[NUM_EVENTS];
    perf_configure_events(event_attrs);

    for (int i = 0; i < NUM_EVENTS; i++) {
        event_fds[i] = perf_event_open(&event_attrs[i], 0, cpu, -1, PERF_FLAG_FD_CLOEXEC);
        assert(event_fds[i] >= 0);
    }
}

void perf_enable(int event_fds[NUM_EVENTS]) {
    for (int i = 0; i < NUM_EVENTS; i++) {
        assert(ioctl(event_fds[i], PERF_EVENT_IOC_RESET, 0) >= 0);
        assert(ioctl(event_fds[i], PERF_EVENT_IOC_ENABLE, 0) >= 0);
    }
}

void perf_disable(int event_fds[NUM_EVENTS]) {
    for (int i = 0; i < NUM_EVENTS; i++) {
        assert(ioctl(event_fds[i], PERF_EVENT_IOC_DISABLE, 0) >= 0);
    }
}

void perf_read(int event_fds[NUM_EVENTS], uint64_t counters[NUM_EVENTS]) {
    for (int i = 0; i < NUM_EVENTS; i++) {
        assert(read(event_fds[i], &counters[i], sizeof(uint64_t)) >= 0);
    }
}

extern memref_descriptor_t spmv(void *, void *, index_t, index_t, index_t,
                                void *, void *, index_t, index_t, index_t,
                                void *, void *, index_t, index_t, index_t,
                                void *, void *, index_t, index_t, index_t,
                                void *, void *, index_t, index_t, index_t);

int main(int argc, char **argv) {

    if (argc < 9) {
        fprintf(stderr, "Usage: %s <rows> <cols> <nnz> <vec> <mat.indices> <mat.indptr> <mat.data> <res>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    uint64_t rows = get_uint64(argv[1]);
    uint64_t cols = get_uint64(argv[2]);
    uint64_t nnz = get_uint64(argv[3]);
    struct shm_info vec = open_and_map_shm(argv[4], true);
    struct shm_info mat_indptr = open_and_map_shm(argv[5], true);
    struct shm_info mat_indices = open_and_map_shm(argv[6], true);
    struct shm_info mat_data = open_and_map_shm(argv[7], true);
    struct shm_info res = open_and_map_shm(argv[8], false);

    memref_descriptor_t res_d = {
        .allocated = res.ptr,
        .aligned = res.ptr,
        .offset = 0,
        .sizes = rows,
        .strides = 1
    };

    memref_descriptor_t mat_indptr_d = {
        .allocated = mat_indptr.ptr,
        .aligned = mat_indptr.ptr,
        .offset = 0,
        .sizes = rows + 1,
        .strides = 1
    };

    memref_descriptor_t mat_indices_d = {
        .allocated = mat_indices.ptr,
        .aligned = mat_indices.ptr,
        .offset = 0,
        .sizes = nnz,
        .strides = 1
    };

    memref_descriptor_t mat_data_d = {
        .allocated = mat_data.ptr,
        .aligned = mat_data.ptr,
        .offset = 0,
        .sizes = nnz,
        .strides = 1
    };

    memref_descriptor_t vec_d = {
        .allocated = vec.ptr,
        .aligned = vec.ptr,
        .offset = 0,
        .sizes = cols,
        .strides = 1
    };

    // Start PMU
    int event_fds[NUM_EVENTS];
    perf_init(event_fds);
    perf_enable(event_fds);

    // Get start timestamp
    struct timespec start_ts;
    clock_gettime(CLOCK_MONOTONIC, &start_ts);

    // Run kernel
    (void)spmv(MEMREF_DESC_ARGS(res_d), MEMREF_DESC_ARGS(mat_indptr_d), MEMREF_DESC_ARGS(mat_indices_d), MEMREF_DESC_ARGS(mat_data_d), MEMREF_DESC_ARGS(vec_d));

    // Get end timestamp
    struct timespec end_ts;
    clock_gettime(CLOCK_MONOTONIC, &end_ts);

    // Stop PMU, read counters
    perf_disable(event_fds);
    uint64_t counters[NUM_EVENTS];
    perf_read(event_fds, counters);

    // Print bench data
    long seconds = end_ts.tv_sec - start_ts.tv_sec;
    long nanoseconds = end_ts.tv_nsec - start_ts.tv_nsec;
    printf("Exec time: %fs\n", seconds + nanoseconds*1e-9);
    printf("inst_retired.any: %lu\n", counters[INST_INDEX]);
    printf("mem_load_uops_retired.l2_hit: %lu\n", counters[L2_HIT_INDEX]);
    printf("mem_load_uops_retired.l3_hit: %lu\n", counters[L3_HIT_INDEX]);
    printf("mem_load_uops_retired.dram_hit: %lu\n", counters[LLC_MISSES_INDEX]);

    // Clean up
    munmap(vec.ptr, vec.sb.st_size);
    munmap(mat_data.ptr, mat_data.sb.st_size);
    munmap(mat_indices.ptr, mat_indices.sb.st_size);
    munmap(mat_indptr.ptr, mat_indptr.sb.st_size);
    munmap(res.ptr, res.sb.st_size);

    return 0;
}
