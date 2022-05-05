#include <iostream>

#include <string>
#include <vector>
#include <complex>

#include <AudioFile.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>

#include "cuda_utilities.cuh"
#include "hann.cuh"
#include "fftw/FFTWPerformer.cuh"
#include "cuda/CUFFTPerformer.cuh"
#include <chrono>
#include <thread>


// adapted from https://stackoverflow.com/a/47785639
const int BYTES_PER_PIXEL = 3; /// red, green, & blue
const int FILE_HEADER_SIZE = 14;
const int INFO_HEADER_SIZE = 40;

void generateBitmapImage(unsigned char* image, int height, int width, char* imageFileName);
unsigned char* createBitmapFileHeader(int height, int stride);
unsigned char* createBitmapInfoHeader(int height, int width);

void generateBitmapImage (unsigned char* image, int height, int width, char* imageFileName)
{
    int widthInBytes = width * BYTES_PER_PIXEL;

    unsigned char padding[3] = {0, 0, 0};
    int paddingSize = (4 - (widthInBytes) % 4) % 4;

    int stride = (widthInBytes) + paddingSize;

    FILE* imageFile = fopen(imageFileName, "wb");

    unsigned char* fileHeader = createBitmapFileHeader(height, stride);
    fwrite(fileHeader, 1, FILE_HEADER_SIZE, imageFile);

    unsigned char* infoHeader = createBitmapInfoHeader(height, width);
    fwrite(infoHeader, 1, INFO_HEADER_SIZE, imageFile);

    int i;
    for (i = 0; i < height; i++) {
        fwrite(image + (i*widthInBytes), BYTES_PER_PIXEL, width, imageFile);
        fwrite(padding, 1, paddingSize, imageFile);
    }

    fclose(imageFile);
}

unsigned char* createBitmapFileHeader (int height, int stride)
{
    int fileSize = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

    static unsigned char fileHeader[] = {
        0,0,     /// signature
        0,0,0,0, /// image file size in bytes
        0,0,0,0, /// reserved
        0,0,0,0, /// start of pixel array
    };

    fileHeader[ 0] = (unsigned char)('B');
    fileHeader[ 1] = (unsigned char)('M');
    fileHeader[ 2] = (unsigned char)(fileSize      );
    fileHeader[ 3] = (unsigned char)(fileSize >>  8);
    fileHeader[ 4] = (unsigned char)(fileSize >> 16);
    fileHeader[ 5] = (unsigned char)(fileSize >> 24);
    fileHeader[10] = (unsigned char)(FILE_HEADER_SIZE + INFO_HEADER_SIZE);

    return fileHeader;
}

unsigned char* createBitmapInfoHeader (int height, int width)
{
    static unsigned char infoHeader[] = {
        0,0,0,0, /// header size
        0,0,0,0, /// image width
        0,0,0,0, /// image height
        0,0,     /// number of color planes
        0,0,     /// bits per pixel
        0,0,0,0, /// compression
        0,0,0,0, /// image size
        0,0,0,0, /// horizontal resolution
        0,0,0,0, /// vertical resolution
        0,0,0,0, /// colors in color table
        0,0,0,0, /// important color count
    };

    infoHeader[ 0] = (unsigned char)(INFO_HEADER_SIZE);
    infoHeader[ 4] = (unsigned char)(width      );
    infoHeader[ 5] = (unsigned char)(width >>  8);
    infoHeader[ 6] = (unsigned char)(width >> 16);
    infoHeader[ 7] = (unsigned char)(width >> 24);
    infoHeader[ 8] = (unsigned char)(height      );
    infoHeader[ 9] = (unsigned char)(height >>  8);
    infoHeader[10] = (unsigned char)(height >> 16);
    infoHeader[11] = (unsigned char)(height >> 24);
    infoHeader[12] = (unsigned char)(1);
    infoHeader[14] = (unsigned char)(BYTES_PER_PIXEL*8);

    return infoHeader;
}

void outputResultsToFile(thrust::host_vector<thrust::host_vector<double> > const& results) {
    thrust::host_vector<double> maxs;
    thrust::transform(results.begin(), results.end(), maxs.begin(), [=] (thrust::host_vector<double> column) {
        return *thrust::max_element(column.begin(), column.end());
    });
    thrust::host_vector<double> mins;
    thrust::transform(results.begin(), results.end(), mins.begin(), [=] (thrust::host_vector<double> column) {
        return *thrust::min_element(column.begin(), column.end());
    });

    double maxOfMaxs = *thrust::max_element(maxs.begin(), maxs.end());
    double maxOfMins = *thrust::max_element(mins.begin(), mins.end());
}

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

    // TODO we need to do image writeback, but how do we figure out the scaling?
    // input is 0 -> 1, so does that mean we end up with a 0 - 1 output? how do we do it at work?
    // for the image, we need to transpose the vector into a flat array, and then invoke the write function
    // for RGB, all we need to do is normalize the results over 0,255 (but how do we do that???)
    // for ^, I think we actually take the min/max of what we have and go with that
    // this code isn't particularly optimal, as it's not part of the evaluated code path

    // TODO get the minMax of the inner vectors using thrust's fany function
    // take each and put them in their own vectors
    // then, reduce on those to get the aboslue max and min, using those values to get the rgb range

    // todo it looks like the actual approach is to take the max of both mins and maxs
    // thrust::host
    thrust::host_vector<double> maxs;
    // thrust::transform()


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