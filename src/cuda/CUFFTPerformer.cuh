#ifndef CUFFTPERFORMER_H_
#define CUFFTPERFORMER_H_

#include <memory>
#include <string>

#include <cufft.h>
#include <AudioFile.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "GPUSamples.cuh"

class CUFFTPerformer
{
private:
    int fft_size;

    int output_fft_size;

    int window_step_size;

    bool complex;

    thrust::device_vector<double> window;

    AudioFile<double> source;

    GPUSamples *in_buffer;

    cufftDoubleComplex *out_buffer;

    cufftHandle plan;

public:
    CUFFTPerformer(int fft_size, AudioFile<double> &source);
    ~CUFFTPerformer();

    thrust::host_vector<thrust::host_vector<double>> performSpectrogram(int startSample, int stopSample);
};

#endif