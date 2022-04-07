#include "CPUSamples.h"

#include <fftw3.h>

CPUSamples::CPUSamples(bool complex, int fft_size)
{
    this->complex = complex;

    if (complex) {
        samples.complex = fftw_alloc_complex(fft_size);
    } else {
        samples.real = fftw_alloc_real(fft_size);
    }
}

CPUSamples::~CPUSamples()
{
    if (complex) {
        fftw_free(samples.complex);
    } else {
        fftw_free(samples.real);
    }
}

bool CPUSamples::isComplex() {
    return complex;
}

CPUSamples::Samples CPUSamples::getSamples() {
    return samples;
}

double* CPUSamples::getReal() {
    return samples.real;
}

fftw_complex* CPUSamples::getComplex() {
    return samples.complex;
}
