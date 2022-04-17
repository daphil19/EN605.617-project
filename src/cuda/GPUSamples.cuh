#ifndef GPUSAMPLES_CUH_
#define GPUSAMPLES_CUH_

# include <vector>

#include <cufft.h>
#include <thrust/device_vector.h>

class GPUSamples
{
private:
    /* data */
    bool complex;
    int size;
    union Samples {
        cufftDoubleReal* real;
        cufftDoubleComplex* complex;
    } samples;
public:
    GPUSamples(bool complex, int fft_size);
    ~GPUSamples();


    // TODO how is all of this cuda stuff supposed to work?

    bool isComplex();
    GPUSamples::Samples getSamples();
    cufftDoubleReal* getReal();
    cufftDoubleComplex* getComplex();
    void clear();
    // TODO are we able to do this? do we need to use thrust?
    void load(std::vector<std::vector<double>> &source, int start, int end);
    void applyWindow(thrust::device_vector<double> window);
};

#endif