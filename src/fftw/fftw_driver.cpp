#include <fftw3.h>

void perfrom_fft()
{
    // these are double-precision; see https://www.fftw.org/fftw3_doc/Complex-numbers.html
    fftw_complex *in, *out;
    fftw_plan p;

    size_t N = 8192;

    // using fftw_malloc is the reccomended approach for allocating data
    in = new fftw_complex[N];
    out = new fftw_complex[N];

    // TODO explain!
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_destroy_plan(p);

    delete[] in;
    delete[] out;
}