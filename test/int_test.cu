#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
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
__attribute__ ((noinline))  void end_roi() {
    usec = (gettime() - usec);
    printf("%d,%d,%lu\n", N, K, usec);
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

void verify(struct GMMParams *params, unsigned *zs, size_t n);

int main(int argc, char **argv) {
    DEBUG = (argc > 1) && (strcmp(argv[1], "--debug") == 0) ? 1 : 0;
    srand(42);

    const unsigned CLASS_MEM_SIZE = K * sizeof(DTYPE),
            PARAM_MEM_SIZE = sizeof(struct GMMParams),
            DATA_MEM_SIZE = N * sizeof(DTYPE),
            ZS_MEM_SIZE = N * sizeof(unsigned);
    unsigned *h_zs = new unsigned[N];
    DTYPE *dataManaged = nullptr;
    struct GmmGibbsState *gibbsState = nullptr;
    struct GMMParams *params = nullptr;

    gpuErrchk(cudaMallocManaged(&params, PARAM_MEM_SIZE));
    gpuErrchk(cudaMallocManaged(&(params->weights), CLASS_MEM_SIZE));
    gpuErrchk(cudaMallocManaged(&(params->means), CLASS_MEM_SIZE));
    gpuErrchk(cudaMallocManaged(&(params->vars), CLASS_MEM_SIZE));
    gpuErrchk(cudaMallocManaged(&(params->zs), ZS_MEM_SIZE));
    gpuErrchk(cudaMallocManaged(&dataManaged, DATA_MEM_SIZE));

    randomInit(dataManaged, h_zs, N, K);
    randInitGmmParams(params, N, K, PRIOR);
    allocGmmGibbsState(&gibbsState, N, K, dataManaged, PRIOR, params);

    begin_roi();
    gibbs(gibbsState, ITERS);
    end_roi();

//    printParams(params, dataManaged, N, K);
//    verify(params, h_zs, N);

    freeGmmGibbsState(gibbsState);
    gpuErrchk(cudaFree(dataManaged));
    delete[] h_zs;
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
    unsigned cat = 0;
    int min = 0, mod = 0;
    for (int i = 0; i < n; i++) {
        if (i < n / k) {
            min = 50;
            mod = 10;
            cat = 0;
        } else if (i < 2*n / k) {
            min = 12;
            mod = 4;
            cat = 1;
        } else if (i < 3*n / k) {
            min = -20;
            mod = 3;
            cat = 2;
        } else {
            min = -90;
            mod = 3;
            cat = 3;
        }
        data[i] = min + (rand() % mod);
        zs[i] = cat;
    }
}

void verify(struct GMMParams *params, unsigned *zs, size_t n) {
    int err = 0;

    for (int i = 0; i < N; i++) {
        if (params->zs[i] != zs[i]) {
            err = 1;
        }
    }

    if (err != 0)
        printf("int_test ---------------------------------------- FAILED! \n");
    else
        printf("int_test ---------------------------------------- SUCCESS! \n");
}