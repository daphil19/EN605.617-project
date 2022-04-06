#ifndef operations_cuh
#define operations_cuh

__host__ cudaEvent_t get_time();
__host__ float get_delta(cudaEvent_t start, cudaEvent_t stop);

#endif
