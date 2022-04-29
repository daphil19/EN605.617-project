#include <iostream>

#include <string>
#include <vector>
#include <complex>

#include <AudioFile.h>

#include "cuda_utilities.cuh"
#include "hann.cuh"
#include "fftw/FFTWPerformer.cuh"
#include "cuda/CUFFTPerformer.cuh"
#include <chrono>
#include <thread>

int main(int argc, char const *argv[])
{
    int fft_size = 8192;

    // the max size we can have for a file (on my 3080) is: 384307168202282325 samples (real)

    // TODO we will need to add more arguments (fftsize, etc.)
    std::string filename = argc == 2 ? argv[2] : "../../sermon.wav"; // TODO args!

    // TODO would it make more sense for me to pass around an instance of the source instead of just the filename

    // give fftw the best shot possible
    fftw_init_threads();
    fftw_plan_with_nthreads(std::thread::hardware_concurrency());

    std::cout << "Attempting to load WAV file " << filename << "... this may take a while" << std::endl;
    FFTWPerformer p(fft_size, filename);
    std::cout << "Successfuly loaded!" << std::endl;
    // cudaEvent_t fftwStart = get_time();
    std::chrono::steady_clock::time_point fftwBegin = std::chrono::steady_clock::now();
    p.performFFT();
    // cudaEvent_t fftwEnd = get_time();
    std::chrono::steady_clock::time_point fftwEnd = std::chrono::steady_clock::now();
    std::cout << "done cpu in: " << std::chrono::duration_cast<std::chrono::milliseconds>(fftwEnd - fftwBegin).count() << std::endl;

    CUFFTPerformer p2(fft_size, filename);
    std::cout << "Beginning the gpu one..." << std::endl;
    cudaEvent_t cufftStart = get_time();
    p2.performFFT();
    cudaEvent_t cufftEnd = get_time();
    std::cout << "done gpu in: " << get_delta(cufftStart, cufftEnd) << std::endl;

    std::cout << "Done!" << std::endl;

    fftw_cleanup();
    fftw_cleanup_threads();

    return EXIT_SUCCESS;
}

/*
0-256
128-384
256-512
384-640
512-768
640-896
768-eof
*/