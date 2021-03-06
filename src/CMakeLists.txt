include(FetchContent)

# include(ExternalProject)
# # use externalProject instead of fetchContent because fftw's cmake configuration is trash
# # (externalProject allows us to customize configure/make/make install whereas fetchcontent assumes cmake targets only)
# ExternalProject_Add(
#     fftw
#     URL http://fftw.org/fftw-3.3.10.tar.gz
#     CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --enable-shared --enable-threads --enable-sse2 --enable-avx --enable-avx2
# )

FetchContent_Declare(
    audiofile
    GIT_REPOSITORY https://github.com/adamstark/AudioFile.git
    GIT_TAG 1.1.0
)

FetchContent_Declare(
    thrust
    GIT_REPOSITORY https://github.com/NVIDIA/thrust.git
    GIT_TAG 1.16.0
)

# FetchContent_Declare(
#     opencv
#     GIT_REPOSITORY https://github.com/opencv/opencv.git
#     GIT_TAG 4.5.5
# )

FetchContent_MakeAvailable(audiofile thrust)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

add_subdirectory(fftw)
add_subdirectory(cuda)

add_executable(en605.617-project main.cu cuda_utilities.cu)

target_link_libraries(en605.617-project fftw_benchmark cuda_benchmark AudioFile)
