#include "GPUSamples.cuh"

#include <thrust/host_vector.h>

#include "kernel_utils.cuh"

// TODO: maybe make these complex kernels a 2-d thing?

__global__ void clear_real(cufftDoubleReal* data) {
    auto idx = get_thread_index();
    data[idx] = 0;
}

__global__ void clear_complex(cufftDoubleComplex* data) {
    auto idx = get_thread_index();
    data[idx].x = 0;
    data[idx].y = 0;
}

__global__ void apply_window_real(cufftDoubleReal* data, double* window) {
    auto idx = get_thread_index();
    data[idx] *= window[idx];
}

__global__ void apply_window_complex(cufftDoubleComplex* data, double* window) {
    auto idx = get_thread_index();
    // auto sample = data[idx];
    data[idx].x *= window[idx];
    data[idx].y *= window[idx];
}

__global__ void normalize_real(cufftDoubleReal* data, double step, int offset) {
    auto idx = get_thread_index();
    data[idx] = (data[idx] + offset) * step;
}

__global__ void normalize_complex(cufftDoubleComplex* data, double step, int offset) {
    auto idx = get_thread_index();
    auto sample = data[idx];
    data[idx].y = (sample.y + offset) * step;
    data[idx].x = (sample.x + offset) * step;
}

GPUSamples::GPUSamples(bool complex, int fft_size)
{
    this->complex = complex;
    size = fft_size;

    // TODO consider malloc host
    // TODO we probably need to fix fft_size
    if (complex) {
        cudaMallocHost((void**) &samples.complex, fft_size * sizeof(cufftDoubleComplex));
    } else {
        cudaMallocHost((void**) &samples.real, fft_size * sizeof(cufftDoubleReal));
    }
}

GPUSamples::~GPUSamples()
{
    if (complex) {
        cudaFreeHost(samples.complex);
    } else {
        cudaFreeHost(samples.real);
    }
}

bool GPUSamples::isComplex() {
    return complex;
}

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
    // TODO WE NEED TO TUNE THE BLOCK AND THREAD SIZES HERE!!!!!!!
    if (complex) {
        // TODO THIS SIZE PROBABLY NEEDS TO CHANGE
        clear_complex<<<1, size>>>(samples.complex);
    } else {
        // TODO THIS SIZE PROBABLY NEEDS TO CHANGE
        clear_real<<<1, size>>>(samples.real);
    }
}

void GPUSamples::load(std::vector<std::vector<double>> &source, int start, int end) {
    // if before the loop this time so that we can appropriately allocate an additional buffer to the 
    if (complex) {
        auto host_buf = new cufftDoubleComplex[end - start];
        // TODO do we need this buffer for both?
        for (int i = 0; i < end - start; i++) {
            host_buf[i].x = source[0][i + start];
            host_buf[i].y = source[1][i + start];
        }

        cudaMemcpy(samples.complex, host_buf, sizeof(cufftDoubleComplex) * (end - start), cudaMemcpyHostToDevice);

        delete[] host_buf;
    } else {
        auto host_buf = new cufftDoubleReal[end - start];
        for (int i = 0; i < end - start; i++) {
            host_buf[i] = source[0][i + start];
        }

        cudaMemcpy(samples.real, host_buf, sizeof(cufftDoubleReal) * (end - start), cudaMemcpyHostToDevice);

        delete[] host_buf;
    }
}

// TODO we might want to be more defensive here, in case fftSize != window size
void GPUSamples::applyWindow(thrust::device_vector<double> window) {
    if (complex) {
        apply_window_complex<<<1, size>>>(samples.complex, thrust::raw_pointer_cast(window.data()));
    } else {
        apply_window_real<<<1, size>>>(samples.real, thrust::raw_pointer_cast(window.data()));
    }
}
