cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(en605.617-project LANGUAGES C CXX CUDA)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif(CCACHE_FOUND)

# globally enable separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(src)
