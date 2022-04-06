# include(FetchContent)

# FetchContent_Declare(
#     gnuradio
#     GIT_REPOSITORY https://github.com/gnuradio/gnuradio.git
#     GIT_TAG v3.10.1.1
# )

# FetchContent_MakeAvailable(gnuradio)

add_subdirectory(fftw)
add_subdirectory(cuda)
add_subdirectory(io)

add_executable(en605.617-project main.cu hann.cpp cuda_utilities.cu)

target_link_libraries(en605.617-project fftw_benchmark cuda_benchmark)
