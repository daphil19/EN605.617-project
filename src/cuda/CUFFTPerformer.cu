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

CUFFTPerformer::CUFFTPerformer(int fft_size, const std::string file)
{
    this->fft_size = fft_size;

    source = AudioFile<double>(file);

    window = hann<thrust::device_vector<double> >(fft_size);

    complex = source.getNumChannels() == 2;

    in_buffer = new GPUSamples(complex, fft_size);

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

void CUFFTPerformer::performFFT() {
    auto num_samples = source.getNumSamplesPerChannel();

    // this results in 50% overlap
    int num_cols = num_samples / (fft_size / 2);


    // TODO this will have to be copied back to the host in some way
    thrust::host_vector<thrust::host_vector<double> > output(num_cols);

    thrust::device_vector<double> col_result(output_fft_size);

    for (int i = 0; i < num_cols; i++) {

        
        // std::cout << i << std::endl;

        // auto cur_col = thrust::host_vector<double>(output_fft_size);

        // std::cout << "pre-clear" << std::endl;

        in_buffer->clear();

        // std::cout << "post-clear" << std::endl;

        // these numbers are in samples!
        auto start = fft_size / 2 * i;
        auto end = std::min(start + fft_size, source.getNumSamplesPerChannel());

        // std::cout << "pre-load" << std::endl;

        // TODO I think that performance improvements here are essential
        // chiefly, I think that batching should attempt to be done if possible
        // though, to be fair, upping the fft_size to a high number really helps 

        in_buffer->load(source.samples, start, end);

        in_buffer->normalize(source.getBitDepth());
        // std::cout << "load done" << std::endl;

        in_buffer->applyWindow(window);

        complex ? cufftExecZ2Z(plan, in_buffer->getComplex(), out_buffer, CUFFT_FORWARD) : cufftExecD2Z(plan, in_buffer->getReal(), out_buffer) ;

        // TODO can we optimize this size?
        post_fft_transform<<<1, output_fft_size>>>(out_buffer, thrust::raw_pointer_cast(col_result.data()));

        // auto out_buf_cast = 

        // TODO i'm thinking of doing the remaining transformations in kernel space, and then copying a resulting device vector back out to the host vector

        // TODO this needs to be kernel-ified as well!
        // for (int j = 0; j < cur_col.size(); j++) {
        //     // cur_col[j] = st::abs(out_buf_cast[j]);
        //     // std::abs(std::)
        //     // TODO we need to take the abs of the thing that we just computed and go from there
        //     cur_col[j] = pow(out_buffer[j].x, 2) + pow(out_buffer[j].y, 2);
        // }

        // NOTE if using vscode, the error squiggle under `__device__` is a false negative; this code compiles fine!
        // thrust::transform(cur_col.begin(), cur_col.end(), cur_col.begin(), [=] (double x) {
        //     double logscale = 10.0 * log10(x);
        //     if (!isfinite(logscale)) {
        //         logscale = MIN_REPLACEMENT;
        //     }
        //     return logscale;
        // });


        auto cur_col = thrust::host_vector<double>(output_fft_size) = col_result;
        // cur_col = col_result;
        output[i] = cur_col;

        // for (int j = 0; j < output_fft_size; j++) {
        //     std::cout << cur_col[j] << " ";
        // }
        // std::cout << "\\" << std::endl;
    }


}