#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <time.h>
#include <vector>
#include "../src/gmm.h"

#ifndef NSAMPLES
#define NSAMPLES (1024)
#endif
#ifndef KCLASSES
#define KCLASSES (4)
#endif

int DEBUG = 1;
const int N = NSAMPLES;
const int K = KCLASSES;
const int ITERS = 500;

static uint64_t usec;

static __inline__ uint64_t gettime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (((uint64_t) tv.tv_sec) * 1000000 + ((uint64_t) tv.tv_usec));
}

__attribute__ ((noinline))  void begin_roi() { usec = gettime(); }

__attribute__ ((noinline))  unsigned long end_roi() {
    return (gettime() - usec);
}

const struct GMMPrior PRIOR = {
        .dirichletPrior=5.0,
        .meansMeanPrior=0.0,
        .meansVarPrior=100.0,
        .varsShapePrior=2.0,
        .varsScalePrior=10.0
};

void printParams(struct GMMParams *params, DTYPE *data, size_t n, size_t k);

void randomInit(DTYPE *data, unsigned *zs, const int n, const int k);

DTYPE accuracy(struct GMMParams *params, unsigned *zs, size_t n);

int main(int argc, char **argv) {
    DEBUG = (argc > 1) && (strcmp(argv[1], "--debug") == 0) ? 1 : 0;
    srand(time(NULL));

    const unsigned CLASS_MEM_SIZE = K * sizeof(DTYPE),
            PARAM_MEM_SIZE = sizeof(struct GMMParams),
            DATA_MEM_SIZE = N * sizeof(DTYPE),
            ZS_MEM_SIZE = N * sizeof(unsigned);
    unsigned *zsTrue = new unsigned[N];
    unsigned long runtime = 0;
    DTYPE *dataManaged = nullptr;
    struct GmmGibbsState *gibbsState = nullptr;
    struct GMMParams *params = nullptr;

    gpuErrchk(cudaMallocManaged(&params, PARAM_MEM_SIZE));
    gpuErrchk(cudaMallocManaged(&(params->weights), CLASS_MEM_SIZE));
    gpuErrchk(cudaMallocManaged(&(params->means), CLASS_MEM_SIZE));
    gpuErrchk(cudaMallocManaged(&(params->vars), CLASS_MEM_SIZE));
    gpuErrchk(cudaMallocManaged(&(params->zs), ZS_MEM_SIZE));
    gpuErrchk(cudaMallocManaged(&dataManaged, DATA_MEM_SIZE));

    randomInit(dataManaged, zsTrue, N, K);
    randInitGmmParams(params, N, K, PRIOR);
    allocGmmGibbsState(&gibbsState, N, K, dataManaged, PRIOR, params);

    begin_roi();
    gibbs(gibbsState, ITERS);
    runtime = end_roi();

//    printParams(params, dataManaged, N, K);
    DTYPE acc = accuracy(params, zsTrue, N);
    printf("%d,%d,%lu,%f\n", N, K, runtime, acc);

    freeGmmGibbsState(gibbsState);
    gpuErrchk(cudaFree(dataManaged));
    delete[] zsTrue;
    return 0;
}

void printParams(struct GMMParams *params, DTYPE *data, size_t n, size_t k) {
    printf("%lu\n", k);
    for (int i = 0; i < k; i++)
        printf("%f %f %f\n", params->weights[i], params->means[i],
               params->vars[i]);
    for (int i = 0; i < n; i++)
        printf("%.2f ", data[i]);
    putchar('\n');
    for (int i = 0; i < n; i++)
        printf("%u ", params->zs[i]);
    putchar('\n');
}

void randomInit(DTYPE *data, unsigned *zs, const int n, const int k) {
    std::vector <std::normal_distribution<DTYPE>> distrs(k);
    std::vector <std::vector<DTYPE>> samples;
    const int N_SAMPLES = 256;
    int means[k], stds[k];

    // Generate arbitrary means and standard deviations
    for (int i = 0; i < k; i++) {
        means[i] = (rand() % n) / k;
        stds[i] = (rand() % 10) + (rand() % 5) - (rand() % 5);
    }

    // Generate distributions, sample N_SAMPLES points, push to dist set
    std::default_random_engine generator;
    for (int i = 0; i < k; i++) {
        std::vector <DTYPE> sample;

        std::normal_distribution <DTYPE> d(means[i], stds[i]);
        for (int j = 0; j < N_SAMPLES; j++)
            sample.push_back(d(generator));

        samples.push_back(sample);
    }

    // Sample from distributions
    for (int i = 0; i < n; i++) {
        int idx = rand() % k;
        data[i] = samples[idx][rand() % N_SAMPLES];
        zs[i] = idx;
    }
}

DTYPE accuracy(struct GMMParams *params, unsigned *zs, size_t n) {
    DTYPE err = 0.0;
    for (int i = 0; i < n; i++)
        if (params->zs[i] != zs[i])
            err += 1.0;
    return 1 - (err / n);
}