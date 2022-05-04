#include "CPUSamples.cuh"

#include <algorithm>
#include <vector>

#include <fftw3.h>
#include <thrust/host_vector.h>

// TODO consider renaming fft_size?
CPUSamples::CPUSamples(bool complex, int fft_size)
{
    this->complex = complex;
    size = fft_size;

    // TODO do we need to worry about a different size?
    if (complex)
    {
        samples.complex = fftw_alloc_complex(fft_size);
    }
    else
    {
        samples.real = fftw_alloc_real(fft_size);
    }
}

CPUSamples::~CPUSamples()
{
    if (complex)
    {
        fftw_free(samples.complex);
    }
    else
    {
        fftw_free(samples.real);
    }
}

bool CPUSamples::isComplex()
{
    return complex;
}

CPUSamples::Samples CPUSamples::getSamples()
{
    return samples;
}

double *CPUSamples::getReal()
{
    return samples.real;
}

fftw_complex *CPUSamples::getComplex()
{
    return samples.complex;
}


void CPUSamples::clear()
{
    // TODO do we want to extract the fill function out?
    if (complex)
    {
        // std::fill feels like a good candidate here, but I couldn't figure out a way to get it to go
        for (int i = 0; i < size; i++) {
            samples.complex[i][0] = 0;
            samples.complex[i][1] = 0;
        }
    }
    else
    {
        std::fill_n(samples.real, size, 0.0);
    }
}

void CPUSamples::load(std::vector<std::vector<double>> &source, int start, int end) {
    for (int i = 0; i < end - start; i++) {
        if (complex) {
            samples.complex[i][0] = source[0][i + start];
            samples.complex[i][1] = source[1][i + start];
        } else {
            samples.real[i] = source[0][i + start];
        }
    }
}

void CPUSamples::applyWindow(thrust::host_vector<double> window) {
    // TODO
    // this function assumes that the window and the buffer are the same size
    for (int i = 0; i < window.size(); i++) {
        if (complex) {
            // TODO how do we compute the window for a complex file?
            // I *think* we treat it like a scalar multiplcation, meaning that we multiply both components
            samples.complex[i][0] *= window[i];
            samples.complex[i][1] *= window[i];
        } else {
            samples.real[i] *= window[i];
        }
    }
}

void CPUSamples::normalize(int bitsPerSample) {
    double step = 1.0 / pow(2, bitsPerSample);
    int offset = pow(2, bitsPerSample - 1) + 1; // intervals are usually [-2^(n - 1) - 1, 2^(n - 1)] for data types
    
    for (int i = 0; i < size; i++) {
        if (complex) {
            samples.complex[i][0] = (samples.complex[i][0] + offset) * step;
            samples.complex[i][1] = (samples.complex[i][1] + offset) * step;
        } else {
            samples.real[i] = (samples.real[i] + offset) * step;
        }
    }
    // if (complex)
}