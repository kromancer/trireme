cmake_minimum_required(VERSION 3.22)

project(baselines LANGUAGES C)

set(L1_MSHRS 10 CACHE STRING "L1 MSHRS")
message(STATUS "L1_MSHRS is set to ${L1_MSHRS}")

set(L2_MSHRS 40 CACHE STRING "L2 MSHRS")
message(STATUS "L2_MSHRS is set to ${L2_MSHRS}")

set(DISABLE_HW_PREF_L1_IPP 0 CACHE STRING "DISABLE HW PREF: L1 IPP ")
message(STATUS "DISABLE_HW_PREF_L1_IPP is set to ${DISABLE_HW_PREF_L1_IPP}, where 1:disabled/0:enabled")

set(DISABLE_HW_PREF_L1_NPP 0 CACHE STRING "DISABLE HW PREF: L1 NPP ")
message(STATUS "DISABLE_HW_PREF_L1_NPP is set to ${DISABLE_HW_PREF_L1_NPP}, where 1:disabled/0:enabled")

set(COMMON_COMPILE_OPTIONS "-g" "-O2")
set(COMMON_COMPILE_DEFS L1_MSHRS=${L1_MSHRS} L2_MSHRS=${L2_MSHRS} DISABLE_HW_PREF_L1_IPP=${DISABLE_HW_PREF_L1_IPP} DISABLE_HW_PREF_L1_NPP=${DISABLE_HW_PREF_L1_NPP})

add_subdirectory(hw_pref_control)

add_executable(spmv-multistage main.c spmv.multistage.c)
add_library(benchmark-spmv-multistage SHARED spmv.multistage.c)

set(TARGETS_LIST spmv-multistage benchmark-spmv-multistage)

foreach(TARGET ${TARGETS_LIST})
    target_compile_options(${TARGET} PRIVATE ${COMMON_COMPILE_OPTIONS})
    target_compile_definitions(${TARGET} PRIVATE ${COMMON_COMPILE_DEFS})
    target_link_libraries(${TARGET} PRIVATE hw_pref_control)
endforeach()
