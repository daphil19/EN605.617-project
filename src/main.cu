#include <iostream>

#include <string>
#include <vector>

#include <AudioFile.h>

#include "cuda_utilities.cuh"
#include "hann.cuh"
#include "fftw/FFTWPerformer.cuh"


int main(int argc, char const *argv[])
{

    // the intent of this driver at this time is to verify that the smart pointer->raw pointer conversion does the right thing

    // double *window;
    // int fft_size = 8192;

    // cudaMalloc((void **)&window, fft_size * sizeof(double));
    // auto window_host = hann(fft_size);
    // // we need to get the actual underlying raw pointer from the smart pointer
    // cudaMemcpy(window, window_host.get(), fft_size * sizeof(double), cudaMemcpyHostToDevice);

    // cudaFree(window);

    // std::vector<std::vector<double>> foo;
    // std::cout << foo.max_size();

    // the max size we can have for a file is: 384307168202282325 samples (that's a lot i think?)


    std::string filename = "../../sermon.wav"; // TODO args!

    // std::cout << "Attempting to load WAV file " << filename << "... this may take a while" << std::endl;
    // AudioFile<double> audiofile(filename);
    // std::cout << "Successfuly loaded!" << std::endl;

    // std::cout << audiofile.getNumSamplesPerChannel() << std::endl;
    // std::cout << audiofile.getNumChannels() << std::endl;

    std::cout << "Attempting to load WAV file " << filename << "... this may take a while" << std::endl;
    FFTWPerformer p(256, "../../sermon.wav");
    std::cout << "Successfuly loaded!" << std::endl;

    p.performFFT();

    // CPUSamples c(true, 8);

    // c.clear();



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