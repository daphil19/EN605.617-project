#ifndef CUFFTPERFORMER_H_
#define CUFFTPERFORMER_H_

#include <memory>

#include "../io/SampleSource.h"

class CUFFTPerformer
{
private:
    /* data */
    // double* input;
    int fft_size;
    // TODO it might make sense to make these smart pointers?

    //////////////////////////////////////
    // TODO WE NEED TO HANDLE COMPLEX DATA (template?)
    //////////////////////////////////////
    // double* window;
    double* data_buffer;
    double* output_buffer;
    // we don't initialize this, but will effectively own it once we get constructed, so make it a smart pointer
    std::shared_ptr<SampleSource> source;

public:
    CUFFTPerformer(int fft_size, std::shared_ptr<SampleSource> source);
    ~CUFFTPerformer();
};


#endif