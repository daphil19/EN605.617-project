#ifndef CPUSAMPLES_CUH_
#define CPUSAMPLES_CUH_

#include <vector>

#include <fftw3.h>
#include <thrust/host_vector.h>

class CPUSamples
{
private:
    bool complex;
    int size;
    union Samples
    {
        double *real;
        fftw_complex *complex;
    } samples;

public:
    CPUSamples(bool complex, int fft_size);
    ~CPUSamples();
    bool isComplex();
    CPUSamples::Samples getSamples();
    double *getReal();
    fftw_complex *getComplex();
    void clear();

    // start and end are expected to be within the bounds... too lazy to error check
    void load(std::vector<std::vector<double>> &source, int start, int end);

    // the window should be the same size as the buffer here
    void applyWindow(thrust::host_vector<double> window);
};

#endif
