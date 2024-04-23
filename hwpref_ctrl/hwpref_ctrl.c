#define _GNU_SOURCE

#include "hwpref_ctrl.h"

#include <assert.h>
#include <sched.h>

#include "hardwarePrefetching/msr.h"

#ifndef DISABLE_HW_PREF_L1_IPP
#warning "DISABLE_HW_PREF_L1_IPP not defined, L1 IPP will be ENABLED (applicapable for only on intel atom cores)"
#define DISABLE_HW_PREF_L1_IPP 0
#endif

#ifndef DISABLE_HW_PREF_L1_NPP
#warning "DISABLE_HW_PREF_L1_NPP not defined, L1 NPP will be ENABLED (applicapable for only on intel atom cores)"
#define DISABLE_HW_PREF_L1_NPP 0
#endif

#ifndef DISABLE_HW_PREF_L2_STREAM
#warning "DISABLE_HW_PREF_L2_STREAM not defined, L2 Stream will be ENABLED (applicapable for only on intel atom cores)"
#define DISABLE_HW_PREF_L2_STREAM 0
#endif

#ifndef DISABLE_HW_PREF_L2_AMP
#warning "DISABLE_HW_PREF_L2_AMP not defined, L2 AMP will be ENABLED (applicapable for only on intel atom cores)"
#define DISABLE_HW_PREF_L2_AMP 0
#endif

#ifndef DISABLE_HW_PREF_LLC_STREAM
#warning "DISABLE_HW_PREF_LLC_STREAM not defined, LLC Stream will be ENABLED (applicapable for only on intel atom cores)"
#define DISABLE_HW_PREF_LLC_STREAM 0
#endif

#if DISABLE_HW_PREF_L1_IPP == 1 || DISABLE_HW_PREF_L1_NPP == 1 || DISABLE_HW_PREF_L2_STREAM == 1 || DISABLE_HW_PREF_L2_AMP == 1 || DISABLE_HW_PREF_LLC_STREAM == 1
#define HW_PREF_CONTROL_ON 1
#else
#define HW_PREF_CONTROL_ON 0
#endif

#define UNITIALIZED (-1)

static msr_t msr;
static int core_id = UNITIALIZED;
static int msr_file = UNITIALIZED;

int init_hw_pref_control(void) {

#if HW_PREF_CONTROL_ON == 1
    core_id = sched_getcpu();

    msr_file = msr_init(core_id, msr);
    if (msr_file < 0)
    {
        return msr_file;
    }
#endif

#if DISABLE_HW_PREF_L1_IPP == 1
    assert(msr_disable_l1ipp(msr) == 0);
#endif

#if DISABLE_HW_PREF_L1_NPP == 1
    assert(msr_disable_l1npp(msr) == 0);
#endif

#if DISABLE_HW_PREF_L2_STREAM == 1
    assert(msr_disable_l2stream(msr) == 0);
#endif

#if DISABLE_HW_PREF_L2_AMP == 1
    assert(msr_disable_l2amp(msr) == 0);
#endif

#if DISABLE_HW_PREF_LLC_STREAM == 1
    assert(msr_disable_llcstream(msr) == 0);
#endif

#if HW_PREF_CONTROL_ON == 1
    assert(msr_write_all(msr_file, msr) >= 0);
#endif

    return 0;
}

void deinit_hw_pref_control(void) {

#if HW_PREF_CONTROL_ON == 1
    assert(core_id > UNITIALIZED);
    assert(msr_file > UNITIALIZED);
    assert(msr_read_all(msr_file, msr) >= 0);
#endif

#if DISABLE_HW_PREF_L1_IPP == 1
    assert(msr_enable_l1ipp(msr) == 1);
#endif

#if DISABLE_HW_PREF_L1_NPP == 1
    assert(msr_enable_l1npp(msr) == 1);
#endif

#if DISABLE_HW_PREF_L2_STREAM == 1
    assert(msr_enable_l2stream(msr) == 1);
#endif

#if DISABLE_HW_PREF_L2_AMP == 1
    assert(msr_enable_l2amp(msr) == 1);
#endif

#if DISABLE_HW_PREF_LLC_STREAM == 1
    assert(msr_enable_llcstream(msr) == 1);
#endif

#if HW_PREF_CONTROL_ON == 1
    assert(msr_write_all(msr_file, msr) >= 0);
#endif

}
