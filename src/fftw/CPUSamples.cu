#include "CPUSamples.cuh"

#include <algorithm>
#include <vector>

#include <fftw3.h>
#include <thrust/host_vector.h>

CPUSamples::CPUSamples(bool complex, int fft_size)
{
    this->complex = complex;
    size = fft_size;

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
    if (complex)
    {
        // std::fill feels like a good candidate here, but I couldn't figure out a way to get it to go
        for (int i = 0; i < size; i++)
        {
            samples.complex[i][0] = 0;
            samples.complex[i][1] = 0;
        }
    }
    else
    {
        std::fill_n(samples.real, size, 0.0);
    }
}

void CPUSamples::load(std::vector<std::vector<double>> &source, int start, int end)
{
    for (int i = 0; i < end - start; i++)
    {
        if (complex)
        {
            samples.complex[i][0] = source[0][i + start];
            samples.complex[i][1] = source[1][i + start];
        }
        else
        {
            samples.real[i] = source[0][i + start];
        }
    }
}

void CPUSamples::applyWindow(thrust::host_vector<double> window)
{
    // this function assumes that the window and the buffer are the same size
    for (int i = 0; i < window.size(); i++)
    {
        if (complex)
        {
            samples.complex[i][0] *= window[i];
            samples.complex[i][1] *= window[i];
        }
        else
        {
            samples.real[i] *= window[i];
        }
    }
}
