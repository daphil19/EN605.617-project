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

// TODO i'm worried that these types are going to fall apart because of the fact that we need complex
// it might make the most sense to have a unioned hierarchy... making things somewhat like a sealed class?
class FFTWPerformer
{
private:
    /* data */
    // double* input;
    int fft_size;
    // TODO it might make sense to make these smart pointers?

    // if we are outputting real, this will be half of the fft_size
    // (this is due to the real-optimized FFT that is performed)
    int output_fft_size;

    int window_step_size;

    bool complex;

    //////////////////////////////////////
    // TODO WE NEED TO HANDLE COMPLEX DATA (template?)
    //////////////////////////////////////
    // std::unique_ptr<double[]> window;
    thrust::host_vector<double> window;
    // std::unique_ptr<double[]> data_buffer;
    fftw_complex* out_buffer;
    // we don't initialize this, but will effectively own it once we get constructed, so make it a smart pointer
    // std::shared_ptr<SampleSource> source;
    AudioFile<double> source;
    std::unique_ptr<CPUSamples> in_buffer;

    fftw_plan plan;

    void normalize();
public:
    // TODO do we want an alternate constructor that provides start/end for the spectrogram?
    FFTWPerformer(int fft_size, AudioFile<double> &source);
    ~FFTWPerformer();

    // void performFFT();
    thrust::host_vector<thrust::host_vector<double> > performFFT();
    
};

#endif