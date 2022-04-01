#include <cmath>

double * hann(int size) {
    double* window = new double[size];
    for (int i = 0; i < size; i++) {
        window[i] = 0.5 * (1 - cos(2*M_PI*i/(size - 1)));
    }

    return window;
}
