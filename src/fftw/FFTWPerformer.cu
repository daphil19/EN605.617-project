#include "FFTWPerformer.cuh"

#include <cmath>
#include <memory>
#include <string>
#include <algorithm>
#include <complex>

#include <fftw3.h>
#include <AudioFile.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>

#include "../io/SampleSource.h"
#include "../hann.cuh"
#include "CPUSamples.cuh"

// TODO maybe the interface is a spectrogramPerformer that has a function that returns a resultant 2-d double array

// this is value is smaller than the log10 of the smallest positive double value
#define MIN_REPLACEMENT -350.0


FFTWPerformer::FFTWPerformer(int fft_size, const std::string file)
{
    this->fft_size = fft_size;
    window = hann<thrust::host_vector<double> >(fft_size);

    // So, this might not work because of the fact that all of the data gets loaded into memory...
    // but the fact that it is a vector means we might be able to get away with a bunch of data? idk...
    source = AudioFile<double>(file);

    // TODO we might not need complex as a member if we can get away with a void pointer that returns the samples naiively
    complex = source.getNumChannels() == 2;

    // this internal output buffer is used so that we can have a single plan, but also emit defensive copies of windows when the code gets executed
    out_buffer = fftw_alloc_complex(fft_size);


    in_buffer = std::unique_ptr<CPUSamples>(new CPUSamples(complex, fft_size));

    window_step_size = fft_size / 2;

    // TODO we might be able to simplify this by providing a void pointer?
    plan = complex ? fftw_plan_dft_1d(fft_size, in_buffer->getComplex(), out_buffer, FFTW_FORWARD, FFTW_ESTIMATE): fftw_plan_dft_r2c_1d(fft_size, in_buffer->getReal(), out_buffer, FFTW_ESTIMATE);
    
}

FFTWPerformer::~FFTWPerformer()
{
    fftw_destroy_plan(plan);
    fftw_free(out_buffer);
}

void FFTWPerformer::performFFT() {

    // TODO is it a window and then an fft, or an fft and then a window?


    // for every fftsize, we will step fftsize / 2
    // we repeat this until we have an fftsize that contains the end of the file

    auto num_samples = source.getNumSamplesPerChannel();


    // the number of colums in the output
    int num_cols = (num_samples / (fft_size / 2)) ;


    // TODO better typing

    // in this structure, the outer index is the column of the image, and the inner index is the row
    // this allows for fft outputs to be bulk copied via memcpy as opposed to iterated over
    // hopefully, we don't need to transpose to get things to work later on

    // TODO WE SHOULD IMPROVE WHAT THE OUTPUT STRUCTURE IS!
    // TODO do we want to convert this to an array? or maybe have it be a thrust vector??
    // std::unique_ptr<std::unique_ptr<double[]>[]> output(new std::unique_ptr<double[]>[num_cols]);

    // TODO will making these references make things faster?
    thrust::host_vector<thrust::host_vector<double> >output(num_cols);

    for (int i = 0; i < num_cols; i++) {
        // first, allocate the results we will be using
        auto cur_col = thrust::host_vector<double>(output_fft_size);
        

        // clear the input buffer in the event we don't have enough data to fill the buffer
        in_buffer->clear();

        // TODO next, we need to load data into the buffer

        // TODO is there a better way for this to be done?
        auto start = fft_size / 2 * i;
        auto end = std::min(start + fft_size, source.getNumSamplesPerChannel());

        // now we _actually_ load the samples
        // source.samples
        in_buffer->load(source.samples, start, end);

        // TODO WE NEED TO DO THINKS LIKE NORMALIZE!

        // window
        in_buffer->applyWindow(window);

        // execute
        fftw_execute(plan);

        // next, we take the magnitude
        // can we get away with shifting to make this faster?

        // write data back to output 
        // TODO how are we supposed to get back to a single data type? is that by taking the magnitude?


        // casting to complex array helps with normalization
        auto out_buf_cast = reinterpret_cast<std::complex<double> *>(out_buffer);
        
        // copy contents into the output, getting the magnitude along the way
        // TODO, also, don't forget about zero samples! this will result in a NaN
        // do we want to just make those the smallest positive double value?
        for (int j = 0; j < cur_col.size(); j++) {
            cur_col[j] = std::abs(out_buf_cast[j]);
            // TODO do we want to handle the zero case here?
        }

        // log scale... 10 log 10
        // thrust::transform(cur_col.begin(), cur_col.end(), thrust::make_constant_iterator(10.0), cur_col.begin(), thrust::multiplies<double>());

        // TODO log
        // TODO do i need host here?
        // log scale... 10 log 10
        thrust::transform(cur_col.begin(), cur_col.end(), cur_col.begin(), [=] (double x) {
            double logscale = 10.0 * log10(x);
            if (isfinite(logscale)) {
                logscale = MIN_REPLACEMENT;
            }
            return logscale;
        });


        // put the results in the output
        output[i] = cur_col;

    }

    /*
        NOTE I think this is the intended order:
        get samples
        normalize
        window
        fft
        results go into pixel
        slide by ...?
     */


}
