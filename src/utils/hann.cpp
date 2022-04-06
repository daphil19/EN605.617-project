#include "hann.h"

#include <cmath>
#include <memory>

std::shared_ptr<double[]> hann(int size) {
    // NOTE for some reason this casues breakage with gcc 11... rolling back to 10 seems to work though?
    std::shared_ptr<double[]> window(new double[size]);
    for (int i = 0; i < size; i++) {
        window[i] = 0.5 * (1 - cos(2*M_PI*i/(size - 1)));
    }

    return window;
}
