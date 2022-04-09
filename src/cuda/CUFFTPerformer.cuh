#ifndef CUFFTPERFORMER_H_
#define CUFFTPERFORMER_H_

#include <memory>

#include <cufft.h>
#include <AudioFile.h>
#include <thrust/device_vector.h>

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

    // TODO out_buffer!

    AudioFile<double> source;

    // TODO in buffer!
    // TODO GPUSamples!

    cufftDoubleComplex* out_buffer;


    cufftHandle plan;

    // double* window;
    // double* data_buffer;
    // double* output_buffer;
    // we don't initialize this, but will effectively own it once we get constructed, so make it a smart pointer
    // std::shared_ptr<SampleSource> source;

public:
    CUFFTPerformer(int fft_size);
    ~CUFFTPerformer();
};


#endif