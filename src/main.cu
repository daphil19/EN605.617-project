#include <iostream>

#include <string>
#include <vector>
#include <complex>

#include <AudioFile.h>

#include "cuda_utilities.cuh"
#include "hann.cuh"
#include "fftw/FFTWPerformer.cuh"
#include "cuda/CUFFTPerformer.cuh"


int main(int argc, char const *argv[])
{

    // the max size we can have for a file (on my 3080) is: 384307168202282325 samples (real)

    // TODO we will need to add more arguments (fftsize, etc.)
    std::string filename = argc == 2 ? argv[2] : "../../sermon.wav"; // TODO args!

    std::cout << "Attempting to load WAV file " << filename << "... this may take a while" << std::endl;
    FFTWPerformer p(256, filename);
    std::cout << "Successfuly loaded!" << std::endl;

    p.performFFT();

    std::cout << "Beginning the gpu one..." << std::endl;

    CUFFTPerformer p2(256, filename);
    p2.performFFT();

    std::cout << "Done!" << std::endl;

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