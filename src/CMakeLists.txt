include(FetchContent)

FetchContent_Declare(
    audiofile
    GIT_REPOSITORY https://github.com/adamstark/AudioFile.git
    GIT_TAG 1.1.0
)

FetchContent_MakeAvailable(audiofile)

add_subdirectory(fftw)
add_subdirectory(cuda)
add_subdirectory(io)
add_subdirectory(utils)

add_executable(en605.617-project main.cu cuda_utilities.cu)

target_link_libraries(en605.617-project fftw_benchmark cuda_benchmark)
