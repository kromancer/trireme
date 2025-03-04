#define _GNU_SOURCE

#include "hwpref_ctrl.h"

#include <assert.h>
#include <sched.h>

#include "hardwarePrefetching/msr.h"

#ifndef DISABLE_HW_PREF_L1_NLP
#warning "DISABLE_HW_PREF_L1_NLP not defined, L1 NLP will be ENABLED (applicable for only on intel atom cores)"
#define DISABLE_HW_PREF_L1_NLP 0
#endif

#ifndef DISABLE_HW_PREF_L1_IPP
#warning "DISABLE_HW_PREF_L1_IPP not defined, L1 IPP will be ENABLED (applicable for only on intel atom cores)"
#define DISABLE_HW_PREF_L1_IPP 0
#endif

#ifndef DISABLE_HW_PREF_L1_NPP
#warning "DISABLE_HW_PREF_L1_NPP not defined, L1 NPP will be ENABLED (applicable for only on intel atom cores)"
#define DISABLE_HW_PREF_L1_NPP 0
#endif

#ifndef DISABLE_HW_PREF_L2_STREAM
#warning "DISABLE_HW_PREF_L2_STREAM not defined, L2 Stream will be ENABLED (applicable for only on intel atom cores)"
#define DISABLE_HW_PREF_L2_STREAM 0
#endif

#ifndef DISABLE_HW_PREF_L2_AMP
#warning "DISABLE_HW_PREF_L2_AMP not defined, L2 AMP will be ENABLED (applicable for only on intel atom cores)"
#define DISABLE_HW_PREF_L2_AMP 0
#endif

#ifndef DISABLE_HW_PREF_LLC_STREAM
#warning "DISABLE_HW_PREF_LLC_STREAM not defined, LLC Stream will be ENABLED (applicable for only on intel atom cores)"
#define DISABLE_HW_PREF_LLC_STREAM 0
#endif

#ifndef SET_L2_STREAM_DD
#warning "SET_L2_STREAM_DD not defined, L2's Demand Density Threshold won't change (applicable for only on intel atom cores)"
#define SET_L2_STREAM_DD -1
#endif


#if (DISABLE_HW_PREF_L1_NLP == 1) || \
  (DISABLE_HW_PREF_L1_IPP == 1) || \
  (DISABLE_HW_PREF_L1_NPP == 1) || \
  (DISABLE_HW_PREF_L2_STREAM == 1) || \
  (DISABLE_HW_PREF_L2_AMP == 1) || \
  (DISABLE_HW_PREF_LLC_STREAM == 1) || \
  (SET_L2_STREAM_DD != -1)
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

    // L2 NLP should be disabled by default
    assert_l2_npl_is_disabled(msr);
#endif

#if DISABLE_HW_PREF_L1_NLP == 1
    assert(msr_disable_l1nlp(msr) == 0);
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

#if SET_L2_STREAM_DD != -1
    msr_set_l2dd(msr, SET_L2_STREAM_DD);
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

#if DISABLE_HW_PREF_L1_NLP == 1
    assert(msr_enable_l1nlp(msr) == 1);
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

#if SET_L2_STREAM_DD != -1
    assert(msr_get_l2dd(msr) == SET_L2_STREAM_DD);
#endif

#if HW_PREF_CONTROL_ON == 1
    assert(msr_write_all(msr_file, msr) >= 0);
#endif

}
