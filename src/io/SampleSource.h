#ifndef IO_SAMPLE_SOURCE_H_
#define IO_SAMPLE_SOURCE_H_

#include <memory>

class SampleSource
{
private:
    /* data */
    bool complex;
public:
    SampleSource(/* args */);
    ~SampleSource();

    // void read(std::unique_ptr<T> buf, int num_samples);

};


#endif