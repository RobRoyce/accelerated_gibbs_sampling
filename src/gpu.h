#pragma once

#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>

// cuRAND state array for uniform distributions
extern __device__ curandState curandStates[];
extern __global__ void setup_kernel();
extern const int N;
extern const int K;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert (N=%d,K=%d): %s %s %d\n", N, K, cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

