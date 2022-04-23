#ifndef hann_cuh_
#define hann_cuh_

#include <cmath>

// in order to make this properly compile the template targets, we have the implementation in the header
template <typename ThrustVectorType>
ThrustVectorType hann(int size) {
    ThrustVectorType window(size);
    for (int i = 0; i < size; i++) {
        window[i] = 0.5 * (1 - cos(2*M_PI*i/(size - 1)));
    }

    return window;
}

#endif