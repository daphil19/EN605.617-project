#ifndef CPUSAMPLES_H_
#define CPUSAMPLES_H_

#include <fftw3.h>

class CPUSamples
{
private:
    /* data */
    bool complex;
    union Samples{
        double* real;
        fftw_complex* complex;
    } samples;
public:
    CPUSamples(bool complex, int fft_size);
    ~CPUSamples();
    bool isComplex();
    // NOTE this might not be needed
    CPUSamples::Samples getSamples();
    double* getReal();
    fftw_complex* getComplex();
};

#endif
