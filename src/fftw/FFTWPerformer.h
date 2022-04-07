#ifndef FFTW_FFTWPERFORMER_H_
#define FFTW_FFTWPERFORMER_H_

#include <memory>
#include <string>

#include <fftw3.h>
#include <AudioFile.h>

#include "../io/SampleSource.h"
#include "CPUSamples.h"


// TODO i'm worried that these types are going to fall apart because of the fact that we need complex
// it might make the most sense to have a unioned hierarchy... making things somewhat like a sealed class?
class FFTWPerformer
{
private:
    /* data */
    // double* input;
    int fft_size;
    // TODO it might make sense to make these smart pointers?

    //////////////////////////////////////
    // TODO WE NEED TO HANDLE COMPLEX DATA (template?)
    //////////////////////////////////////
    std::unique_ptr<double[]> window;
    // std::unique_ptr<double[]> data_buffer;
    fftw_complex* output_buffer;
    // we don't initialize this, but will effectively own it once we get constructed, so make it a smart pointer
    // std::shared_ptr<SampleSource> source;
    AudioFile<double> source;
    std::unique_ptr<CPUSamples> data;
public:
    FFTWPerformer(int fft_size, const std::string file);
    ~FFTWPerformer();

    void PerformFFT();
};

#endif