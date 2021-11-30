#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <time.h>
#include <vector>
#include <assert.h>
#include "../src/gmm.h"

#ifndef NSAMPLES
#define NSAMPLES (1024)
#endif
#ifndef KCLASSES
#define KCLASSES (4)
#endif
#ifndef NITERS
#define NITERS (500)
#endif

int DEBUG = 1;
const int N = NSAMPLES;
const int K = KCLASSES;
const int ITERS = NITERS;
__device__ curandState curandStates[NSAMPLES];

__global__ void call_uniform(DTYPE a, DTYPE b, DTYPE *res) {
    *res = uniform(a, b);
}

void test_uniform() {
    // Initialize CUDA random states
    setupKernel<<<3, 1>>>();
    gpuErrchk(cudaDeviceSynchronize());

    std::vector <std::pair<int, int>> uniformParams = {
            {0,  0},
            {1,  1},
            {-1, 1},
            {-9, 3}
    };
    std::vector <DTYPE> res;
    DTYPE *d, *h = (DTYPE *) malloc(uniformParams.size());
    cudaMallocManaged(&d, uniformParams.size());

    for (int i = 0; i < uniformParams.size(); i++) {
        call_uniform<<<1, 1>>>(uniformParams[i].first, uniformParams[i].second, &d[i]);
    }

    cudaMemcpy(h, d, uniformParams.size() * sizeof(DTYPE), cudaMemcpyDeviceToHost);

    for (int i = 0; i < uniformParams.size(); i++) {
        printf("%f ", h[i]);
    }
    printf("\n");


    assert(res[0] == 0.000000);
    assert(res[1] == 1.000000);
//    assert(res[2] == 1.000000);
//    assert(res[3] <= 2 && res[3] >= 0);
//    assert(res[4] == -99999999999);
}

int main(int argc, char **argv) {
    test_uniform();
}

