add_library(cuda_benchmark cuda_driver.cu)
target_link_libraries(cuda_benchmark cufft)
