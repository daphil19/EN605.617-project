#include "kernel_utils.cuh"

__device__ unsigned int get_thread_index()
{
    return (blockIdx.x * blockDim.x) + threadIdx.x;
}
