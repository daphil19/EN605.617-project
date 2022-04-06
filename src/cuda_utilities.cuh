#ifndef cuda_utilities_cuh
#define cuda_utilities_cuh

__host__ cudaEvent_t get_time();
__host__ float get_delta(cudaEvent_t start, cudaEvent_t stop);

__device__ unsigned int get_thread_index();

#endif
