#include "FFTWPerformer.h"

#include <memory>

#include <fftw3.h>

#include "../io/SampleSource.h"
#include "../utils/hann.h"

FFTWPerformer::FFTWPerformer(int fft_size, std::shared_ptr<SampleSource> source)
{
    this->fft_size = fft_size;
    this->source = source;
    window = hann(fft_size);
    data_buffer = new double[fft_size];
    // this internal output buffer is used so that we can have a single plan, but also emit defensive copies of windows when the code gets executed
    // TODO is sizing correct?
    output_buffer = new double[fft_size];
}

FFTWPerformer::~FFTWPerformer()
{
    delete[] output_buffer;
    delete[] data_buffer;
}

void FFTWPerformer::PerformFFT() {

}