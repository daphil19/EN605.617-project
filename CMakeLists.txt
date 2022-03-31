cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(en605.617-project LANGUAGES C CXX CUDA)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif(CCACHE_FOUND)

include(FetchContent)
include(ExternalProject)

# use externalProject instead of fetchContent because fftw's cmake configuration is trash
# (externalProject allows us to customize configure/make/make install whereas fetchcontent assumes cmake targets only)
ExternalProject_Add(
    fftw
    URL http://fftw.org/fftw-3.3.10.tar.gz
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --enable-shared --enable-threads 
)

FetchContent_Declare(
    audiofile
    GIT_REPOSITORY https://github.com/adamstark/AudioFile.git
    GIT_TAG 1.1.0
)

set(FFTW_ENABLE_THREADS ON)

FetchContent_MakeAvailable(audiofile)

# globally enable separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(en605.617-project fftw_driver.cpp)

# manually identify the fftw installation
ExternalProject_Get_Property(fftw INSTALL_DIR)
target_include_directories(en605.617-project PRIVATE ${INSTALL_DIR}/include)
target_link_directories(en605.617-project PRIVATE ${INSTALL_DIR}/lib)
target_link_libraries(en605.617-project PRIVATE AudioFile fftw3_threads fftw3 m)

add_dependencies(en605.617-project fftw)
