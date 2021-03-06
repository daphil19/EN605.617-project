#include "FFTWPerformer.cuh"

#include <cmath>
#include <memory>
#include <string>
#include <algorithm>
#include <complex>

#include <iostream>

#include <fftw3.h>
#include <AudioFile.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>

#include "../hann.cuh"
#include "CPUSamples.cuh"

// this is value is smaller than the log10 of the smallest positive double value
#define MIN_REPLACEMENT -350.0

FFTWPerformer::FFTWPerformer(int fft_size, AudioFile<double> &source)
{
    this->fft_size = fft_size;
    window = hann<thrust::host_vector<double>>(fft_size);

    this->source = source;

    complex = source.getNumChannels() == 2;

    out_buffer = fftw_alloc_complex(fft_size);

    in_buffer = std::unique_ptr<CPUSamples>(new CPUSamples(complex, fft_size));

    window_step_size = fft_size / 2;

    plan = complex ? fftw_plan_dft_1d(fft_size, in_buffer->getComplex(), out_buffer, FFTW_FORWARD, FFTW_ESTIMATE) : fftw_plan_dft_r2c_1d(fft_size, in_buffer->getReal(), out_buffer, FFTW_ESTIMATE);

    output_fft_size = complex ? fft_size : (fft_size / 2) + 1;
}

FFTWPerformer::~FFTWPerformer()
{
    fftw_destroy_plan(plan);
    fftw_free(out_buffer);
}

void FFTWPerformer::normalize()
{
    double step = 1.0 / source.getBitDepth();
    int offset = pow(2, source.getBitDepth() - 1) + 1; // intervals are usually [-2^(n - 1) - 1, 2^(n - 1)] for data types
}

thrust::host_vector<thrust::host_vector<double>> FFTWPerformer::performSpectrogram(int startSample, int stopSample)
{
    // for every fftsize, we will step fftsize / 2
    // we repeat this until we have an fftsize that contains the end of the file

    // TODO we are assuming that the start and stop samples passed in are valid values based on the source
    int num_samples = stopSample - startSample;

    // the number of colums in the output
    // we are aiming for a 50% overlap
    int num_cols = num_samples / (fft_size / 2);

    // in this structure, the outer index is the column of the image, and the inner index is the row
    // this allows for fft outputs to be bulk copied via memcpy as opposed to iterated over
    thrust::host_vector<thrust::host_vector<double>> output(num_cols);

    for (int i = 0; i < num_cols; i++)
    {

        // first, allocate the results we will be using
        auto cur_col = thrust::host_vector<double>(output_fft_size);

        // clear the input buffer in the event we don't have enough data to fill the buffer
        in_buffer->clear();

        // these numbers are in samples!
        auto start = startSample + fft_size / 2 * i;
        auto end = std::min(start + fft_size, num_samples);

        // now we _actually_ load the samples
        in_buffer->load(source.samples, start, end);

        // window
        in_buffer->applyWindow(window);

        // execute
        fftw_execute(plan);

        // next, we take the magnitude (squared)
        // casting to complex array helps with normalization
        auto out_buf_cast = reinterpret_cast<std::complex<double> *>(out_buffer);

        // this one seems to be the fastest, as we don't have an extra copy and still benefit from some of the transform being parallel
        for (int j = 0; j < cur_col.size(); j++)
        {
            cur_col[j] = std::abs(out_buf_cast[j]);
        }
        thrust::transform(cur_col.begin(), cur_col.end(), cur_col.begin(), [=](double x)
                          {
            double logscale = 10.0 * log10(pow(x, 2));
            if (!isfinite(logscale)) {
                logscale = MIN_REPLACEMENT;
            }
            return logscale; });

        // this approach may be slower,
        // BUT, with the optimizations turned on it actually ends up being pretty fast...
        // auto out_buf_vec = thrust::host_vector<std::complex<double>>(out_buf_cast, out_buf_cast + output_fft_size);
        // thrust::transform(out_buf_vec.begin(), out_buf_vec.end(), cur_col.begin(), [=] (std::complex<double> x) {
        //     double magSquared = pow(std::abs(x), 2);
        //     double logScale = 10.0 * log10(magSquared);
        //     return isfinite(logScale) ? logScale : MIN_REPLACEMENT;
        // });

        // put the results in the output
        output[i] = cur_col;
    }

    return output;
}
