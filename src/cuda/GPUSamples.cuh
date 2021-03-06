#ifndef GPUSAMPLES_CUH_
#define GPUSAMPLES_CUH_

#include <vector>

#include <cufft.h>
#include <thrust/device_vector.h>

class GPUSamples
{
private:
    bool complex;
    int size;
    union Samples
    {
        cufftDoubleReal *real;
        cufftDoubleComplex *complex;
    } samples;

public:
    GPUSamples(bool complex, int fft_size);
    ~GPUSamples();

    bool isComplex();
    GPUSamples::Samples getSamples();
    cufftDoubleReal *getReal();
    cufftDoubleComplex *getComplex();
    void clear();
    void load(std::vector<std::vector<double>> &source, int start, int end);
    void applyWindow(thrust::device_vector<double> window);
};

#endif