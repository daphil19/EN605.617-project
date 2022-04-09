cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(en605.617-project LANGUAGES CXX CUDA)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif(CCACHE_FOUND)

# globally enable separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# used so that we can have lambdas on the device (or host)
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

add_subdirectory(src)
