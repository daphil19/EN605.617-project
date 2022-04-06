#include "cuda_utilities.cuh"

#include "utils/hann.h"

int main(int argc, char const *argv[])
{

    // the intent of this driver at this time is to verify that the smart pointer->raw pointer conversion does the right thing

    double* window;
    int fft_size = 8192;

    cudaMalloc((void**) &window, fft_size * sizeof(double));
    auto window_host = hann(fft_size);
    // we need to get the actual underlying raw pointer from the smart pointer
    cudaMemcpy(window, window_host.get(), fft_size * sizeof(double), cudaMemcpyHostToDevice);

    cudaFree(window);

    return 0;
}
