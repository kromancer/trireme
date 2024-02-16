cmake_minimum_required(VERSION 3.25)

project(runahead-with-omp-tasks LANGUAGES C)

if(NOT DEFINED PREFETCH_DISTANCE)
    set(PREFETCH_DISTANCE 32)
endif()

set(LLVM_PATH $ENV{LLVM_PATH})
message(STATUS "LLVM_PATH: ${LLVM_PATH}")

# assert that LLVM_PATH points to llvm's build directory
set(OMP_HEADER "${LLVM_PATH}/projects/openmp/runtime/src/omp.h")
if (EXISTS "${OMP_HEADER}")
    message(STATUS "Found OpenMP header: ${OMP_HEADER}")
else()
    message(FATAL_ERROR "Could not locate OpenMP header, is LLVM_PATH set correctly? Was llvm's OpenMP built?")
endif()

message(STATUS "prefetch distance (PD) is set to ${PREFETCH_DISTANCE}")

set(COMMON_INCLUDE_DIRS "${LLVM_PATH}/projects/openmp/runtime/src")
set(COMMON_COMPILE_OPTIONS "-std=c23" "-fopenmp" "-DPD=${PREFETCH_DISTANCE}")
set(COMMON_LINK_OPTIONS "-L${LLVM_PATH}/lib" "-lomp" "-Wl,-rpath,${LLVM_PATH}/lib")

add_executable(test-spmv-runahead spmv-runahead-with-omp-tasks.c main.c)
target_include_directories(test-spmv-runahead PRIVATE ${COMMON_INCLUDE_DIRS})
target_compile_options(test-spmv-runahead PRIVATE ${COMMON_COMPILE_OPTIONS})
target_link_options(test-spmv-runahead PRIVATE ${COMMON_LINK_OPTIONS})

add_library(spmv-runahead SHARED spmv-runahead-with-omp-tasks.c)
target_include_directories(spmv-runahead PRIVATE ${COMMON_INCLUDE_DIRS})
target_compile_options(spmv-runahead PRIVATE ${COMMON_COMPILE_OPTIONS})
target_link_options(spmv-runahead PRIVATE ${COMMON_LINK_OPTIONS})
