#include "hw_pref_control.h"

#include <assert.h>
#include <sched.h>

#include "msr.h"

#ifndef DISABLE_HW_PREF_L1_IPP
#warning "DISBLE_HW_PREF_L1_IPP not defined, L1 IPP will be ENABLED (applicapable for only on intel atom cores)"
#define DISABLE_HW_PREF_L1_IPP 0
#endif

#ifndef DISABLE_HW_PREF_L1_NPP
#warning "DISBLE_HW_PREF_L1_NPP not defined, L1 NPP will be ENABLED (applicapable for only on intel atom cores)"
#define DISABLE_HW_PREF_L1_NPP 0
#endif

static union msr_u msr[HWPF_MSR_FIELDS];
static int core_id;
static int msr_file;

void init_hw_pref_control(void) {

#if DISABLE_HW_PREF_L1_IPP == 1 || DISABLE_HW_PREF_L1_NPP == 1
    int core_id = sched_getcpu();
    int msr_file = msr_int(core_id, msr);
#endif

#if DISABLE_HW_PREF_L1_IPP == 1
    assert(msr_disable_l1ipp(msr) == 0);
#endif

#if DISABLE_HW_PREF_L1_NPP == 1
    assert(msr_disable_l1npp(msr) == 0);
#endif

#if DISABLE_HW_PREF_L1_IPP == 1 || DISABLE_HW_PREF_L1_NPP == 1
    assert(msr_hwpf_write(msr_file, msr) > 0);
#endif

}

void deinit_hw_pref_control(void) {

#if DISABLE_HW_PREF_L1_IPP == 1 || DISABLE_HW_PREF_L1_NPP == 1
    assert(msr_hwpf_read(msr_file, msr) > 0);
#endif

#if DISABLE_HW_PREF_L1_IPP == 1
    assert(msr_enable_l1ipp(msr) == 1);
#endif

#if DISABLE_HW_PREF_L1_NPP == 1
    assert(msr_enable_l1npp(msr) == 1);
#endif

#if DISABLE_HW_PREF_L1_IPP == 1 || DISABLE_HW_PREF_L1_NPP == 1
    assert(msr_hwpf_write(msr_file, msr) > 0);
#endif

}
