#define _GNU_SOURCE
#include <sched.h>

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
#include <x86intrin.h>
#include <errno.h>

#define CYCLES_INDEX 0
#define INST_INDEX 1
#define ALL_LOADS_INDEX 2
#define LLC_MISSES_INDEX 3
#define NUM_EVENTS 4

long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu,
                            int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

// Configure perf events
void perf_configure_events(struct perf_event_attr *pea) {
    static const uint64_t e_cfg[NUM_EVENTS][2] = {
        [CYCLES_INDEX] = {0x00c0, 10},
        [INST_INDEX] = {0x00c2, 10},
        [ALL_LOADS_INDEX] = {0x81d0, 10},
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
        event_fds[i] = perf_event_open(&event_attrs[i], 0, -1, -1, PERF_FLAG_FD_CLOEXEC);
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

int main(int argc, char **argv) {

    int event_fds[NUM_EVENTS];
    perf_init(event_fds);
    perf_enable(event_fds);

    // Dummy workload to trigger loads + cycles
    #define N 10000
    volatile uint64_t x[N];
    uint64_t sum = 0;
    for (int i = 0; i < N; i++) { sum += x[i]; }

    perf_disable(event_fds);

    uint64_t counters[NUM_EVENTS];
    perf_read(event_fds, counters);

    printf("cpu_clk_unhalted.core: %lu\n", counters[CYCLES_INDEX]);
    printf("inst_retired.any: %lu\n", counters[INST_INDEX]);
    printf("mem_uops_retired.all_loads: %lu\n", counters[ALL_LOADS_INDEX]);
    printf("mem_load_uops_retired.dram_hit: %lu\n", counters[LLC_MISSES_INDEX]);

    return 0;
}
