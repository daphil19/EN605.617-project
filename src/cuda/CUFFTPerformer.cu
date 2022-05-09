#include "CUFFTPerformer.cuh"

#include <memory>
#include <string>
#include <cmath>

#include <iostream>

#include <AudioFile.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>

#include "GPUSamples.cuh"
#include "kernel_utils.cuh"
#include "../hann.cuh"

// TODO can we put this in a shared location?
#define MIN_REPLACEMENT -350.0

// TODO magnitude and log scale
__global__ void post_fft_transform(cufftDoubleComplex* data, double* output) {
    auto idx = get_thread_index();
    cufftDoubleComplex sample = data[idx];
    // TODO do i need to worry about doubles here?
    double magSquare = pow(sample.x, 2) + pow(sample.y, 2);
    double logScale = 10.0 * log10(magSquare);
    // potential warping here! taking the branch should be pretty rare though
    output[idx] = isfinite(logScale) ? logScale : MIN_REPLACEMENT;
}

CUFFTPerformer::CUFFTPerformer(int fft_size, AudioFile<double> &source)
{
    this->fft_size = fft_size;

    this->source = source;

    window = hann<thrust::device_vector<double> >(fft_size);

    complex = source.getNumChannels() == 2;

    in_buffer = new GPUSamples(complex, fft_size);

    // this doesn't need to be page locked because it never interacts with anlything on the host
    cudaMalloc((void**)&out_buffer, fft_size * sizeof(cufftDoubleComplex));

    cufftPlan1d(&plan, fft_size, complex ? CUFFT_Z2Z : CUFFT_D2Z, 1);

    output_fft_size = complex ? fft_size : (fft_size / 2) + 1;

}

CUFFTPerformer::~CUFFTPerformer()
{
    cufftDestroy(plan);
    cudaFree(out_buffer);
    delete in_buffer;
}

thrust::host_vector<thrust::host_vector<double> > CUFFTPerformer::performSpectrogram(int startSample, int stopSample) {
    auto num_samples = stopSample - startSample;

    // this results in 50% overlap
    int num_cols = num_samples / (fft_size / 2);


    // TODO this will have to be copied back to the host in some way
    thrust::host_vector<thrust::host_vector<double> > output(num_cols);

    thrust::device_vector<double> col_result(output_fft_size);

    for (int i = 0; i < num_cols; i++) {

                in_buffer->clear();

        // these numbers are in samples!
        auto start = startSample + fft_size / 2 * i;
        auto end = std::min(start + fft_size, num_samples);


        // TODO I think that performance improvements here are essential
        // chiefly, I think that batching should attempt to be done if possible
        // though, to be fair, upping the fft_size to a high number really helps 

        in_buffer->load(source.samples, start, end);

        in_buffer->applyWindow(window);

        cufftResult res = complex ? cufftExecZ2Z(plan, in_buffer->getComplex(), out_buffer, CUFFT_FORWARD) : cufftExecD2Z(plan, in_buffer->getReal(), out_buffer) ;

        cudaDeviceSynchronize();

        // TODO can we optimize this size?
        post_fft_transform<<<1, output_fft_size>>>(out_buffer, thrust::raw_pointer_cast(col_result.data()));

        auto cur_col = thrust::host_vector<double>(output_fft_size) = col_result;
        output[i] = cur_col;
    }

    return output;
}