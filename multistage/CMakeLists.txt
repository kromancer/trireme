cmake_minimum_required(VERSION 3.22)

project(baselines LANGUAGES C)

set(L1_MSHRS 10 CACHE STRING "L1 MSHRS")
message(STATUS "L1_MSHRS is set to ${L1_MSHRS}")

set(L2_MSHRS 40 CACHE STRING "L2 MSHRS")
message(STATUS "L2_MSHRS is set to ${L2_MSHRS}")

option(PROFILE_WITH_VTUNE "Enable profiling with VTune" OFF)

add_executable(spmv-multistage main.c spmv.multistage.c)
add_library(benchmark-spmv-multistage SHARED spmv.multistage.c)
set(TARGETS_LIST spmv-multistage benchmark-spmv-multistage)

foreach(TARGET ${TARGETS_LIST})
    target_compile_options(${TARGET} PRIVATE "-g" "-O2")
    target_compile_definitions(${TARGET} PRIVATE
        L1_MSHRS=${L1_MSHRS}
        L2_MSHRS=${L2_MSHRS})

    if(PROFILE_WITH_VTUNE)
        target_compile_definitions(${TARGET} PRIVATE PROFILE_WITH_VTUNE=1)
        target_include_directories(${TARGET} PRIVATE /opt/intel/oneapi/vtune/latest/sdk/include)
        target_link_directories(${TARGET} PRIVATE /opt/intel/oneapi/vtune/latest/sdk/lib64)
        target_link_libraries(${TARGET} PRIVATE ittnotify)
    endif()

endforeach()
