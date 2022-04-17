#include "GPUSamples.cuh"

#include <thrust/host_vector.h>

GPUSamples::GPUSamples(bool complex, int fft_size)
{
    this->complex = complex;
    size = fft_size;

    // TODO we probably need to fix fft_size
    if (complex) {
        cudaMalloc((void**) &samples.complex, fft_size * sizeof(cufftDoubleComplex));
    } else {
        cudaMalloc((void**) &samples.real, fft_size * sizeof(cufftDoubleReal));
    }
}

GPUSamples::~GPUSamples()
{
    if (complex) {
        cudaFree(samples.complex);
    } else {
        cudaFree(samples.real);
    }
}

bool GPUSamples::isComplex() {
    return complex;
}

// TODO we have to do the rest of these

GPUSamples::Samples GPUSamples::getSamples() {
    return samples;
}

cufftDoubleReal* GPUSamples::getReal() {
    return samples.real;
}

cufftDoubleComplex* GPUSamples::getComplex() {
    return samples.complex;
}

void GPUSamples::clear() {

    if (complex) {
        // TODO is there a better way to do this in complex?
        for (int i = 0; i < size; i++) {
            samples.complex[i].x = 0;
            samples.complex[i].y = 0;
        }
    }
    // flush all bytes of all double values to zero
    // TODO are we allowed to do this?
    cudaMemset(samples.real, 0, sizeof(cufftDoubleReal) * size);
}

// // TODO are we able to do this? do we need to use thrust?
// void GPUSamples::load(std::vector<std::vector<double>> &source, int start, int end);

void GPUSamples::applyWindow(thrust::device_vector<double> window) {
    for (int i = 0; i < window.size(); i++) {
        if (complex) {
            samples.complex[i].x *= window[i];
            samples.complex[i].y *= window[i];
        } else {
            samples.real[i] *= window[i];
        }
    }
}
