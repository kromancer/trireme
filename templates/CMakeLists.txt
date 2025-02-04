cmake_minimum_required(VERSION 3.22)

project(spmv LANGUAGES C)

# What's the index bitwidth?
set(INDEX_TYPE "int32" CACHE STRING "Set index type: int32 or int64")

# Convert the string to a numerical macro value
if(INDEX_TYPE STREQUAL "int32")
    set(INDEX_BITWIDTH 32)
    set(INDEX_TYPE_DEF "uint32_t")
elseif(INDEX_TYPE STREQUAL "int64")
    set(INDEX_BITWIDTH 64)
    set(INDEX_TYPE_DEF "uint64_t")
else()
    message(FATAL_ERROR "Invalid INDEX_TYPE: Use int32 or int64")
endif()
message(STATUS "INDEX_TYPE set to ${INDEX_TYPE_DEF}, INDEX_BITWIDTH=${INDEX_BITWIDTH}")

# Use OpenMP from llvm
#
# Search for the OpenMP header in the LLVM_PATH
find_path(OMP_HEADER_PATH
        NAMES "omp.h"
        PATHS $ENV{LLVM_PATH}/projects/openmp/runtime/src $ENV{LLVM_PATH}/lib/clang/20/include
        NO_DEFAULT_PATH)

if (NOT OMP_HEADER_PATH)
    message(FATAL_ERROR "Could not locate OpenMP header, is LLVM_PATH pointing to either installation or build dir?")
else()
    message(STATUS "Found OpenMP header in: ${OMP_HEADER_PATH}")
endif()

# Search for the OpenMP library
find_library(OMP_LIBRARY
        NAMES "omp" "libomp"
        PATHS $ENV{LLVM_PATH}/lib $ENV{LLVM_PATH}/lib/x86_64-unknown-linux-gnu/
        NO_DEFAULT_PATH)

if(OMP_LIBRARY)
    get_filename_component(OMP_LIB_PATH "${OMP_LIBRARY}" DIRECTORY)
    message(STATUS "Found OpenMP library in: ${OMP_LIB_PATH}")
else()
    message(FATAL_ERROR "OpenMP library not found")
endif()

if(NOT DEFINED KERNEL_LLVM_IR)
    message(FATAL_ERROR "KERNEL_LLVM_IR must be specified to locate the LLVM IR file.")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(USE_AVX2_FLAG "-mavx2")
else()
    set(USE_AVX2_FLAG "")
endif()

add_custom_command(
    OUTPUT kernel.o
    COMMAND ${CMAKE_C_COMPILER} -c ${USE_AVX2_FLAG} -O3 -Wno-override-module -o kernel.o ${KERNEL_LLVM_IR}
    DEPENDS ${KERNEL_LLVM_IR}
)

add_library(kernel STATIC kernel.o)
set_target_properties(kernel PROPERTIES LINKER_LANGUAGE C)

if(NOT DEFINED MAIN_FILE)
    message(FATAL_ERROR "MAIN_FILE must be specified to locate the main source file.")
endif()

add_executable(main ${MAIN_FILE} utils.c)
target_link_libraries(main PRIVATE kernel)

target_compile_options(main PRIVATE
 -fopenmp
 -g
 -O0
 -DINDEX_BITWIDTH=${INDEX_BITWIDTH}
 )

target_link_options(main PRIVATE
 -L${OMP_LIB_PATH}
 -lomp
 -Wl,-rpath,${OMP_LIB_PATH}
 )
