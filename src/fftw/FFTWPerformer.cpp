#include "FFTWPerformer.h"

#include <memory>
#include<string>

#include <fftw3.h>
#include <AudioFile.h>

#include "../io/SampleSource.h"
#include "../utils/hann.h"
#include "CPUSamples.h"

// TODO maybe the interface is a spectrogramPerformer that has a function that returns a resultant 2-d double array



FFTWPerformer::FFTWPerformer(int fft_size, const std::string file)
{
    this->fft_size = fft_size;
    window = hann(fft_size);

    // std::unique_ptr<double[]> data_buffer(new double[fft_size]);
    // this internal output buffer is used so that we can have a single plan, but also emit defensive copies of windows when the code gets executed
    // TODO is sizing correct?
    output_buffer = fftw_alloc_complex(fft_size);

    source = AudioFile<double>(file);

    // CPUSamples test = CPUSamples(true, 3);

    data = std::unique_ptr<CPUSamples>(new CPUSamples(true, 5));

    
}

FFTWPerformer::~FFTWPerformer()
{
    fftw_free(output_buffer);
}

void FFTWPerformer::PerformFFT() {

}