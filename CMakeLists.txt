cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(en605.617-project LANGUAGES C CXX CUDA)

# enabling & configuring one of these might help if there are issues finding fftw3
# in theory, these can also be appended when calling cmake as well
# see: https://stackoverflow.com/a/44489201
#set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}:/path/to/fftw/pkgconfig")
#set(ENV{PKG_CONFIG_PATH} "${ENV{PKG_CONFIG_PATH}}:/path/to/fftw/pkgconfig")

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
include_directories(PkgConfig::FFTW)
link_libraries     (PkgConfig::FFTW)


add_executable(en605.617-project fftw_driver.c)

set_target_properties(
    en605.617-project
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

