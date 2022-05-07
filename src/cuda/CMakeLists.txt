add_library(cuda_benchmark CUFFTPerformer.cu GPUSamples.cu)
target_link_libraries(cuda_benchmark cufft AudioFile Thrust m)
