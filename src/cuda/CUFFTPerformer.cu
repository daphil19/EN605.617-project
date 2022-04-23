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
#include "../hann.cuh"

// TODO can we put this in a shared location?
#define MIN_REPLACEMENT -350.0


CUFFTPerformer::CUFFTPerformer(int fft_size, const std::string file)
{
    this->fft_size = fft_size;

    source = AudioFile<double>(file);

    window = hann<thrust::device_vector<double> >(fft_size);

    complex = source.getNumChannels() == 2;

    in_buffer = new GPUSamples(complex, fft_size);

    cudaMalloc((void**)&out_buffer, fft_size * sizeof(cufftDoubleComplex));

    cufftPlan1d(&plan, fft_size, complex ? CUFFT_Z2Z : CUFFT_D2Z, 1);
}

CUFFTPerformer::~CUFFTPerformer()
{
    cufftDestroy(plan);
    cudaFree(out_buffer);
    delete in_buffer;
}

void CUFFTPerformer::performFFT() {
    auto num_samples = source.getNumSamplesPerChannel();

    // TODO is this supposed to be / 2 iff the data is real instead of at all times?
    auto num_cols = (num_samples / (fft_size / 2));


    // TODO this will have to be copied back to the host in some way
    thrust::host_vector<thrust::host_vector<double> > output(num_cols);

    for (int i = 0; i < num_cols; i++) {
        
        // std::cout << i << std::endl;

        auto cur_col = thrust::host_vector<double>(output_fft_size);

        std::cout << "pre-clear" << std::endl;

        in_buffer->clear();

        std::cout << "post-clear" << std::endl;

        auto start = fft_size / 2 * i;
        auto end = std::min(start + fft_size, source.getNumSamplesPerChannel());

        std::cout << "pre-load" << std::endl;

        in_buffer->load(source.samples, start, end);

        in_buffer->applyWindow(window);

        complex ? cufftExecZ2Z(plan, in_buffer->getComplex(), out_buffer, CUFFT_FORWARD) : cufftExecD2Z(plan, in_buffer->getReal(), out_buffer) ;

        // auto out_buf_cast = 

        // TODO this needs to be kernel-ified as well!
        for (int j = 0; j < cur_col.size(); j++) {
            // cur_col[j] = st::abs(out_buf_cast[j]);
            // std::abs(std::)
            // TODO we need to take the abs of the thing that we just computed and go from there
            cur_col[j] = pow(out_buffer[j].x, 2) + pow(out_buffer[j].y, 2);
        }

        // NOTE if using vscode, the error squiggle under `__device__` is a false negative; this code compiles fine!
        thrust::transform(cur_col.begin(), cur_col.end(), cur_col.begin(), [=] __device__ (double x) {
            double logscale = 10.0 * log10(x);
            if (isfinite(logscale)) {
                logscale = MIN_REPLACEMENT;
            }
            return logscale;
        });

        output[i] = cur_col;

    }


    // TODO: WE NEED TO COPY THE DATA BACK ONTO THE HOST IN SOME WAY!!!!!!!!!!!!!!!!


}