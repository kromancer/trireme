cmake_minimum_required(VERSION 3.22)

project(hwpref_ctrl_wrapper LANGUAGES C)

set(DISABLE_HW_PREF_L1_IPP 0 CACHE STRING "DISABLE HW PREF: L1 IPP ")
message(STATUS "DISABLE_HW_PREF_L1_IPP is set to ${DISABLE_HW_PREF_L1_IPP}, where 1:disabled/0:enabled")

set(DISABLE_HW_PREF_L1_NPP 0 CACHE STRING "DISABLE HW PREF: L1 NPP ")
message(STATUS "DISABLE_HW_PREF_L1_NPP is set to ${DISABLE_HW_PREF_L1_NPP}, where 1:disabled/0:enabled")

set(DISABLE_HW_PREF_L2_STREAM 0 CACHE STRING "DISABLE HW PREF: L2 Stream ")
message(STATUS "DISABLE_HW_PREF_L2_STREAM is set to ${DISABLE_HW_PREF_L2_STREAM}, where 1:disabled/0:enabled")

set(DISABLE_HW_PREF_L2_AMP 0 CACHE STRING "DISABLE HW PREF: L2 AMP ")
message(STATUS "DISABLE_HW_PREF_L2_AMP is set to ${DISABLE_HW_PREF_L2_AMP}, where 1:disabled/0:enabled")

set(DISABLE_HW_PREF_LLC_STREAM 0 CACHE STRING "DISABLE HW PREF: LLC Stream ")
message(STATUS "DISABLE_HW_PREF_LLC_STREAM is set to ${DISABLE_HW_PREF_LLC_STREAM}, where 1:disabled/0:enabled")

add_subdirectory(hardwarePrefetching)

# hwpref_control_wrapper only on x86_64
add_library(hwpref_ctrl_wrapper SHARED
    $<IF:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>, hwpref_ctrl.c, hwpref_ctrl_dummy.c>)
target_link_libraries(hwpref_ctrl_wrapper PRIVATE
    $<$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>:hwpref_ctrl>)

target_compile_options(hwpref_ctrl_wrapper PRIVATE "-g" "-O0")
target_compile_definitions(hwpref_ctrl_wrapper PRIVATE
    DISABLE_HW_PREF_L1_IPP=${DISABLE_HW_PREF_L1_IPP}
    DISABLE_HW_PREF_L1_NPP=${DISABLE_HW_PREF_L1_NPP}
    DISABLE_HW_PREF_L2_STREAM=${DISABLE_HW_PREF_L2_STREAM}
    DISABLE_HW_PREF_L2_AMP=${DISABLE_HW_PREF_L2_AMP}
    DISABLE_HW_PREF_LLC_STREAM=${DISABLE_HW_PREF_LLC_STREAM})
