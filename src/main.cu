#include <iostream>

#include <string>
#include <vector>
#include <complex>

#include <AudioFile.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>

#include "cuda_utilities.cuh"
#include "hann.cuh"
#include "fftw/FFTWPerformer.cuh"
#include "cuda/CUFFTPerformer.cuh"
#include <chrono>
#include <thread>

const int NUM_ITERATIONS = 10;
const int MIN_FFT_POWER = 8;
const int MAX_FFT_POWER = 20;

// adapted from https://stackoverflow.com/a/47785639
const int BYTES_PER_PIXEL = 3; /// red, green, & blue
const int FILE_HEADER_SIZE = 14;
const int INFO_HEADER_SIZE = 40;

// i could (maybe should?) refactor all of the unsigned char* refs to std::byte, but not right now
unsigned char *createBitmapFileHeader(int height, int stride)
{
    int fileSize = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

    static unsigned char fileHeader[] = {
        0, 0,       /// signature
        0, 0, 0, 0, /// image file size in bytes
        0, 0, 0, 0, /// reserved
        0, 0, 0, 0, /// start of pixel array
    };

    fileHeader[0] = (unsigned char)('B');
    fileHeader[1] = (unsigned char)('M');
    fileHeader[2] = (unsigned char)(fileSize);
    fileHeader[3] = (unsigned char)(fileSize >> 8);
    fileHeader[4] = (unsigned char)(fileSize >> 16);
    fileHeader[5] = (unsigned char)(fileSize >> 24);
    fileHeader[10] = (unsigned char)(FILE_HEADER_SIZE + INFO_HEADER_SIZE);

    return fileHeader;
}

unsigned char *createBitmapInfoHeader(int height, int width)
{
    static unsigned char infoHeader[] = {
        0, 0, 0, 0, /// header size
        0, 0, 0, 0, /// image width
        0, 0, 0, 0, /// image height
        0, 0,       /// number of color planes
        0, 0,       /// bits per pixel
        0, 0, 0, 0, /// compression
        0, 0, 0, 0, /// image size
        0, 0, 0, 0, /// horizontal resolution
        0, 0, 0, 0, /// vertical resolution
        0, 0, 0, 0, /// colors in color table
        0, 0, 0, 0, /// important color count
    };

    infoHeader[0] = (unsigned char)(INFO_HEADER_SIZE);
    infoHeader[4] = (unsigned char)(width);
    infoHeader[5] = (unsigned char)(width >> 8);
    infoHeader[6] = (unsigned char)(width >> 16);
    infoHeader[7] = (unsigned char)(width >> 24);
    infoHeader[8] = (unsigned char)(height);
    infoHeader[9] = (unsigned char)(height >> 8);
    infoHeader[10] = (unsigned char)(height >> 16);
    infoHeader[11] = (unsigned char)(height >> 24);
    infoHeader[12] = (unsigned char)(1);
    infoHeader[14] = (unsigned char)(BYTES_PER_PIXEL * 8);

    return infoHeader;
}

void generateBitmapImage(unsigned char *image, int height, int width, char *imageFileName)
{
    int widthInBytes = width * BYTES_PER_PIXEL;

    unsigned char padding[3] = {0, 0, 0};
    int paddingSize = (4 - (widthInBytes) % 4) % 4;

    int stride = (widthInBytes) + paddingSize;

    FILE *imageFile = fopen(imageFileName, "wb");

    unsigned char *fileHeader = createBitmapFileHeader(height, stride);
    fwrite(fileHeader, 1, FILE_HEADER_SIZE, imageFile);

    unsigned char *infoHeader = createBitmapInfoHeader(height, width);
    fwrite(infoHeader, 1, INFO_HEADER_SIZE, imageFile);

    int i;
    for (i = 0; i < height; i++)
    {
        fwrite(image + (i * widthInBytes), BYTES_PER_PIXEL, width, imageFile);
        fwrite(padding, 1, paddingSize, imageFile);
    }

    fclose(imageFile);
}

void outputResultsToFile(thrust::host_vector<thrust::host_vector<double>> const &results, char *outfile)
{
    thrust::host_vector<double> maxs(results.size());
    thrust::transform(results.begin(), results.end(), maxs.begin(), [=](thrust::host_vector<double> column)
                      { return *thrust::max_element(column.begin(), column.end()); });
    thrust::host_vector<double> mins(results.size());
    thrust::transform(results.begin(), results.end(), mins.begin(), [=](thrust::host_vector<double> column)
                      { return *thrust::min_element(column.begin(), column.end()); });

    double maxOfMaxs = *thrust::max_element(maxs.begin(), maxs.end());
    double maxOfMins = *thrust::max_element(mins.begin(), mins.end());

    size_t height = results[0].size();
    size_t width = results.size();

    thrust::host_vector<thrust::host_vector<unsigned char>> bytes(results.size(), thrust::host_vector<unsigned char>(results[0].size()));

    for (int i = 0; i < results.size(); i++)
    {
        thrust::transform(results[i].begin(), results[i].end(), bytes[i].begin(), [=](double res)
                          {
            if (res <= maxOfMins) {
                return (unsigned char)255;
            } else if (res >= maxOfMaxs) {
                return (unsigned char)0;
            } else {
                return (unsigned char)round( 255 * (res - maxOfMins) / (maxOfMaxs - maxOfMins));
            } });
    }

    unsigned char imageBytes[height][width][BYTES_PER_PIXEL];
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int k = 0; k < BYTES_PER_PIXEL; k++)
            {
                imageBytes[i][j][k] = bytes[j][i];
            }
        }
    }

    generateBitmapImage((unsigned char *)imageBytes, height, width, outfile);
}

void verifySpectrogramOutputs()
{
    int fft_size = 256;

    AudioFile<double> source("../../testing123-mono.wav");
    FFTWPerformer fftw(fft_size, source);
    auto fftwResults = fftw.performSpectrogram(0, source.getNumSamplesPerChannel());
    outputResultsToFile(fftwResults, (char *)"../../fftw-results.bmp");

    CUFFTPerformer cufft(fft_size, source);
    auto cufftResults = cufft.performSpectrogram(0, source.getNumSamplesPerChannel());
    outputResultsToFile(fftwResults, (char *)"../../cufft-results.bmp");
}

void performBenchmark(std::string filename, std::vector<int> const &sampleCounts, thrust::host_vector<thrust::host_vector<double>> &results)
{
    AudioFile<double> source(filename);
    for (int i = MIN_FFT_POWER; i <= MAX_FFT_POWER; i++)
    {
        int fft_size = pow(2, i);
        std::cout << "using fft size " << fft_size << std::endl;
        for (int j = 0; j < sampleCounts.size(); j++)
        {
            int stopSample = sampleCounts[j];
            std::cout << "using sample count " << stopSample << std::endl;
            thrust::host_vector<double> fftw_times;
            thrust::host_vector<double> cufft_times;
            for (int k = 0; k < NUM_ITERATIONS; k++)
            {
                std::cout << "iteration " << k + 1 << " of " << NUM_ITERATIONS << std::endl;
                FFTWPerformer p(fft_size, source);
                std::chrono::steady_clock::time_point fftwBegin = std::chrono::steady_clock::now();
                std::cout << "performing cpu" << std::endl;
                auto results = p.performSpectrogram(0, stopSample);
                std::chrono::steady_clock::time_point fftwEnd = std::chrono::steady_clock::now();
                fftw_times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(fftwEnd - fftwBegin).count());

                CUFFTPerformer p2(fft_size, source);
                cudaEvent_t cufftStart = get_time();
                std::cout << "performing gpu" << std::endl;
                auto results2 = p2.performSpectrogram(0, stopSample);
                cudaEvent_t cufftEnd = get_time();
                cufft_times.push_back(get_delta(cufftStart, cufftEnd));
            }

            int fftw_sum = thrust::reduce(fftw_times.begin(), fftw_times.end());
            int cufft_sum = thrust::reduce(cufft_times.begin(), cufft_times.end());

            thrust::host_vector<double> fftw;
            thrust::host_vector<double> cufft;
            fftw.push_back(fft_size);

            fftw.push_back(stopSample);

            fftw.push_back(fftw_sum / NUM_ITERATIONS);
            fftw.push_back(cufft_sum / NUM_ITERATIONS);

            results.push_back(fftw);
        }
    }
}

int main(int argc, char const *argv[])
{
    bool verify = argc == 2 && std::string("--verify").compare(argv[1]) == 0;

    fftw_init_threads();
    fftw_plan_with_nthreads(std::thread::hardware_concurrency());

    std::cout << "fftw performer will use up to " << fftw_planner_nthreads() << " threads" << std::endl;

    if (verify)
    {
        std::cout << "Running spectrogram verification code paths." << std::endl;
        verifySpectrogramOutputs();
    }
    else
    {
        int fullDataSize = AudioFile<double>("../../sermon.wav").getNumSamplesPerChannel();
        std::vector sampleCounts{50000, /*100000, 1000000, 10000000, fullDataSize*/};

        std::cout << "Performing benchmark" << std::endl;
        thrust::host_vector<thrust::host_vector<double>> complex_results;
        performBenchmark("../../sermon.wav", sampleCounts, complex_results);

        thrust::host_vector<thrust::host_vector<double>> real_results;
        performBenchmark("../../sermon-mono.wav", sampleCounts, real_results);

        std::cout << "fft size | data size | fftw time | cufft time" << std::endl;
        std::cout << "COMPLEX RESULTS:" << std::endl;
        for (int i = 0; i < complex_results.size(); i++)
        {
            auto resultRow = complex_results[i];
            for (int j = 0; j < resultRow.size(); j++)
            {
                std::cout << resultRow[j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "REAL RESULTS:" << std::endl;
        for (int i = 0; i < real_results.size(); i++)
        {
            auto resultRow = real_results[i];
            for (int j = 0; j < resultRow.size(); j++)
            {
                std::cout << resultRow[j] << " ";
            }
            std::cout << std::endl;
        }
    }

    fftw_cleanup();
    fftw_cleanup_threads();

    std::cout << "Done!" << std::endl;
    return EXIT_SUCCESS;
}
