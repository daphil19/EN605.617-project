#ifndef FFTW_FFTWPERFORMER_H_
#define FFTW_FFTWPERFORMER_H_

#include "FFTWPerformer.h"

#include <memory>

#include <fftw3.h>

#include "../io/SampleSource.h"


class FFTWPerformer
{
private:
    /* data */
    // double* input;
    int fft_size;
    // TODO it might make sense to make these smart pointers?
    double* window;
    double* data_buffer;
    double* output_buffer;
    // we don't initialize this, but will effectively own it once we get constructed, so make it a smart pointer
    std::shared_ptr<SampleSource> source;
public:
    FFTWPerformer(int fft_size, std::shared_ptr<SampleSource> source);
    ~FFTWPerformer();

    void PerformFFT();
};

#endif