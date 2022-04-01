add_subdirectory(fftw)
add_subdirectory(cuda)

add_executable(en605.617-project main.cpp)

target_link_libraries(en605.617-project fftw_benchmark cuda_benchmark)
