cmake_minimum_required(VERSION 3.22)

project(runahead-with-omp-tasks LANGUAGES C)

set(VARIANT "omp-tasks" CACHE STRING "variant of the spmv-runahead source file")
set(SPMV_RUNAHEAD_SRC "spmv-runahead-${VARIANT}.c")
message(STATUS "Using ${SPMV_RUNAHEAD_SRC}")

set(PREFETCH_DISTANCE 32 CACHE STRING "prefetch distance")
message(STATUS "Prefetch distance is set to ${PREFETCH_DISTANCE}")

set(LOCALITY_HINT 3 CACHE STRING "locality hint")
message(STATUS "The locality hint is set to ${LOCALITY_HINT}")

# LLVM_PATH can refer to the llvm installation OR build directory
set(LLVM_PATH $ENV{LLVM_PATH})
message(STATUS "LLVM_PATH: ${LLVM_PATH}")

option(ENABLE_LOGS "Enable logging to stdout with printfs" OFF)
if(ENABLE_LOGS)
    list(APPEND COMMON_COMPILE_DEFS "ENABLE_LOGS")
    message(STATUS "logging is enabled")
endif()

# Use OpenMP from llvm
#
# Search for the OpenMP header in the LLVM_PATH
find_path(OMP_HEADER_PATH
        NAMES "omp.h"
        PATHS ${LLVM_PATH}/projects/openmp/runtime/src ${LLVM_PATH}/lib/clang/19/include
        NO_DEFAULT_PATH)

if (NOT OMP_HEADER_PATH)
    message(FATAL_ERROR "Could not locate OpenMP header, is LLVM_PATH pointing to either installation or build dir?")
else()
    message(STATUS "Found OpenMP header in: ${OMP_HEADER_PATH}")
endif()

# Search for the OpenMP library
find_library(OMP_LIBRARY
        NAMES "omp" "libomp"
        PATHS "${LLVM_PATH}/lib" "${LLVM_PATH}/lib/x86_64-unknown-linux-gnu/"
        NO_DEFAULT_PATH)

if(OMP_LIBRARY)
    get_filename_component(OMP_LIB_PATH "${OMP_LIBRARY}" DIRECTORY)
    message(STATUS "Found OpenMP library in: ${OMP_LIB_PATH}")
else()
    message(FATAL_ERROR "OpenMP library not found")
endif()

set(COMMON_INCLUDE_DIRS "${OMP_HEADER_PATH}")
set(COMMON_COMPILE_OPTIONS "-fopenmp" "-g" "-O2")
set(COMMON_COMPILE_DEFS PD=${PREFETCH_DISTANCE} LOCALITY_HINT=${LOCALITY_HINT})
set(COMMON_LINK_OPTIONS "-L${OMP_LIB_PATH}" "-lomp" "-Wl,-rpath,${OMP_LIB_PATH}")

add_executable(test-spmv-runahead ${SPMV_RUNAHEAD_SRC} test.c)
add_library(benchmark-spmv-runahead SHARED ${SPMV_RUNAHEAD_SRC})
add_executable(spmv-runahead ${SPMV_RUNAHEAD_SRC} main.c)

set(TARGETS_LIST test-spmv-runahead benchmark-spmv-runahead spmv-runahead)

foreach(TARGET ${TARGETS_LIST})
    target_include_directories(${TARGET} PRIVATE ${COMMON_INCLUDE_DIRS})
    target_compile_options(${TARGET} PRIVATE ${COMMON_COMPILE_OPTIONS})
    target_compile_definitions(${TARGET} PRIVATE ${COMMON_COMPILE_DEFS})
    target_link_options(${TARGET} PRIVATE ${COMMON_LINK_OPTIONS})
endforeach()
