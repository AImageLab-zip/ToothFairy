#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <algorithm>
#include <cstdio>


#define MAX_THREADS 1024
#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))


inline int optThreads(const int work) {
    const int pow_2 = std::log((double)work) / std::log(2.0);
    return std::max(std::min(1 << pow_2, MAX_THREADS), 1);
}


inline dim3 optThreads(const int x, const int y) {
    const int xThreads = optThreads(x);
    const int yThreads = std::max(std::min(optThreads(y), MAX_THREADS / xThreads), 1);
    dim3 threads(xThreads, yThreads);
    return threads;
}


#define DEBUG 0
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      std::fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#endif
