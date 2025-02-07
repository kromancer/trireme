cmake_minimum_required(VERSION 3.22)

set(NUM_OF_STREAMS 0 CACHE STRING "NUMBER_OF_STREAMS ")
message(STATUS "NUM_OF_STREAMS is set to ${NUM_OF_STREAMS}")

project(spmv LANGUAGES C)

add_executable(one_page_per_stream one_page_per_stream.c)
target_compile_options(one_page_per_stream PRIVATE -O1 -g -D{NUM_OF_STREAMS})

add_executable(one_page_many_streams one_page_many_streams.c)
target_compile_options(one_page_many_streams PRIVATE -O1 -g -D{NUM_OF_STREAMS})

