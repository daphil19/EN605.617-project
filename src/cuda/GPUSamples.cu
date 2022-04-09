#include "GPUSamples.cuh"

GPUSamples::GPUSamples(/* args */)
{
}

GPUSamples::~GPUSamples()
{
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
    // TODO
}

// // TODO are we able to do this? do we need to use thrust?
// void GPUSamples::load(std::vector<std::vector<double>> &source, int start, int end);
