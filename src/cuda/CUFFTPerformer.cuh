#ifndef CUFFTPERFORMER_H_
#define CUFFTPERFORMER_H_

#include <memory>
#include <string>

#include <cufft.h>
#include <AudioFile.h>
#include <thrust/device_vector.h>

#include "GPUSamples.cuh"

class CUFFTPerformer
{
private:
    /* data */
    // double* input;
    int fft_size;

    int output_fft_size;

    int window_step_size;

    bool complex;

    thrust::device_vector<double> window;

    AudioFile<double> source;

    // TODO in buffer!
    // TODO GPUSamples!
    GPUSamples* in_buffer;

    cufftDoubleComplex* out_buffer;


    cufftHandle plan;

    // double* window;
    // double* data_buffer;
    // double* output_buffer;
    // we don't initialize this, but will effectively own it once we get constructed, so make it a smart pointer
    // std::shared_ptr<SampleSource> source;

public:
    CUFFTPerformer(int fft_size, const std::string file);
    ~CUFFTPerformer();

    void performFFT();
};


#endif