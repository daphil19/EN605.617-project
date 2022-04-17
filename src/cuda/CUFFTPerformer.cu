#include "CUFFTPerformer.cuh"

#include <memory>
#include <string>

#include <AudioFile.h>
#include <thrust/device_vector.h>

#include "GPUSamples.cuh"
#include "../hann.cuh"

CUFFTPerformer::CUFFTPerformer(int fft_size, const std::string file)
{
    this->fft_size = fft_size;

    source = AudioFile<double>(file);

    window = hann<thrust::device_vector<double> >(fft_size);

    complex = source.getNumChannels() == 2;

    in_buffer = new GPUSamples(complex, fft_size);

    cudaMalloc((void**)&out_buffer, fft_size * sizeof(cufftDoubleComplex));

    cufftPlan1d(&plan, fft_size, complex ? CUFFT_Z2Z : CUFFT_D2Z, 1);

    // this->source = source;
    // the window can simply live on the gpu, and since we only transfer to the
    // host during construction time, it can safely be paged
    // cudaMalloc((void**) &window, fft_size * sizeof(double));
    // auto window_host = hann(fft_size);
    // we need to get the actual underlying raw pointer from the smart pointer
    // cudaMemcpy(window, window_host.get(), fft_size * sizeof(double), cudaMemcpyHostToDevice);

    // both inputs and outputs will likely need to interact with the host, so make these page locked
    // cudaMallocHost((void**) &data_buffer, fft_size * sizeof(double));
    // cudaMallocHost((void**) &output_buffer, fft_size * sizeof(double));

    // window = new double[fft_size];
    // data_buffer = new double[fft_size];
    // // this internal output buffer is used so that we can have a single plan, but also emit defensive copies of windows when the code gets executed
    // // TODO is sizing correct?
    // output_buffer = new double[fft_size];    
}

CUFFTPerformer::~CUFFTPerformer()
{
    cufftDestroy(plan);
    cudaFree(out_buffer);
    delete in_buffer;
    // cudaFree(output_buffer)?;
    // cudaFree(data_buffer);
    // cudaFree(window);
}

void CUFFTPerformer::performFFT() {

}