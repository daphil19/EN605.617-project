#include <fftw3.h>

#include <AudioFile.h>

int main()
{
    // these are double-precision; see https://www.fftw.org/fftw3_doc/Complex-numbers.html
    fftw_complex *in, *out;
    fftw_plan p;

    size_t N = 8192;

    // using fftw_malloc is the reccomended approach for allocating data
    in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);

    // TODO explain!
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_destroy_plan(p);

    fftw_free(in);
    fftw_free(out);

    return 0;
}