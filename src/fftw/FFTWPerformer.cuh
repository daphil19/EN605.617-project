#ifndef FFTW_FFTWPERFORMER_H_
#define FFTW_FFTWPERFORMER_H_

#include <memory>
#include <string>

#include <fftw3.h>
#include <AudioFile.h>
#include <thrust/host_vector.h>

#include "CPUSamples.cuh"

// This header (and it's implementation) are .cu files so that nvcc could be used to link thrust
// all code and data structures are run on the host

class FFTWPerformer
{
private:
    int fft_size;

    // if we are outputting real, this will be half of the fft_size
    // (this is due to the real-optimized FFT that is performed)
    int output_fft_size;

    int window_step_size;

    bool complex;

    thrust::host_vector<double> window;

    fftw_complex *out_buffer;

    AudioFile<double> source;

    std::unique_ptr<CPUSamples> in_buffer;

    fftw_plan plan;

    void normalize();

public:
    FFTWPerformer(int fft_size, AudioFile<double> &source);
    ~FFTWPerformer();
    thrust::host_vector<thrust::host_vector<double>> performSpectrogram(int startSample, int stopSample);
};

#endif