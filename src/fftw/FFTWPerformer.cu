#include "FFTWPerformer.cuh"

#include <memory>
#include <string>
#include <algorithm>

#include <fftw3.h>
#include <AudioFile.h>
#include <thrust/host_vector.h>

#include "../io/SampleSource.h"
#include "../hann.cuh"
#include "CPUSamples.cuh"

// TODO maybe the interface is a spectrogramPerformer that has a function that returns a resultant 2-d double array



FFTWPerformer::FFTWPerformer(int fft_size, const std::string file)
{
    this->fft_size = fft_size;
    window = hann<thrust::host_vector<double> >(fft_size);

    // TODO we might not need complex as a member if we can get away with a void pointer that returns the samples naiively
    complex = source.getNumChannels() == 2;

    // this internal output buffer is used so that we can have a single plan, but also emit defensive copies of windows when the code gets executed
    out_buffer = fftw_alloc_complex(fft_size);

    // So, this might not work because of the fact that all of the data gets loaded into memory...
    // but the fact that it is a vector means we might be able to get away with a bunch of data? idk...
    source = AudioFile<double>(file);

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
    std::unique_ptr<std::unique_ptr<double[]>[]> output(new std::unique_ptr<double[]>[num_cols]);

    for (int i = 0; i < num_cols; i++) {
        // first, allocate the results we will be using
        output[i] = std::unique_ptr<double[]>(new double[fft_size]);

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
        // in_buffer->applyWindow(window);

        // execute
        fftw_execute(plan);

        // write data back to output 
    }
    



    // std::unique_ptr<double[][]> output(new double[fft_size * 2][fft_size]);

    // std::unique_ptr<std::unique_ptr<

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
