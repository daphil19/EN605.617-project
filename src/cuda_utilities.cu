#include "cuda_utilities.cuh"

__host__ cudaEvent_t get_time()
{
    cudaEvent_t time;
    cudaEventCreate(&time);
    cudaEventRecord(time);
    return time;
}

__host__ float get_delta(cudaEvent_t start, cudaEvent_t stop)
{
    cudaEventSynchronize(stop);

    float delta = 0;
    cudaEventElapsedTime(&delta, start, stop);
    return delta;
}

__device__ unsigned int get_thread_index()
{
    return (blockIdx.x * blockDim.x) + threadIdx.x;
}
