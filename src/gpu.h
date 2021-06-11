#pragma once

#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// cuRAND state array for uniform distributions
extern __device__ curandState curandStates[];

extern __global__ void setup_kernel();
