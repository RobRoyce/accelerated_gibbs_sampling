#pragma once

#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include "math_constants.h"

#ifndef NSAMPLES
#define NSAMPLES (1024)
#endif
#ifndef KCLASSES
#define KCLASSES (16)
#endif
#ifndef M_E
# define M_E 2.7182818284590452354 /* e */
#endif
#ifndef M_PI
# define M_PI 3.14159265358979323846 /* pi */
#endif


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert [%d/%d]: %s %s %d\n", NSAMPLES, KCLASSES, cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// cuRAND state array for uniform distributions
extern __device__ curandState curandStates[];

extern __global__ void setup_kernel();
