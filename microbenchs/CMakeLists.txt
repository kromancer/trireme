cmake_minimum_required(VERSION 3.22)

project(spmv LANGUAGES C)

add_executable(hw_prefs hw_prefs.c)

target_compile_options(hw_prefs PRIVATE -O1 -g)
