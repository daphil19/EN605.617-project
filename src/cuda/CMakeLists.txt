add_library(cuda_benchmark cuda_driver.cu CUFFTPerformer.cu GPUSamples.cu)
target_link_libraries(cuda_benchmark cufft AudioFile)
