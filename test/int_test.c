#include "../src/gmm.h"
#include <vector>
#include <iostream>
#include <random>

#ifndef NSAMPLES
#define NSAMPLES (1024)
#endif
#ifndef KCLASSES
#define KCLASSES (4)
#endif

int DEBUG = 1;
const int N = NSAMPLES;
const int K = KCLASSES;
const int ITERS = 10000;


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

void printParams(struct gmm_params *params, DTYPE *data, size_t n, size_t k);

void randomInit(DTYPE *data, unsigned *zs, const int n, const int k);

DTYPE accuracy(struct gmm_params *params, unsigned *zs, size_t n);

const struct gmm_prior PRIOR = {
        .dirichlet_prior=5.0f,
        .means_mean_prior=0.0f,
        .means_var_prior=100.0f,
        .vars_shape_prior=2.0f,
        .vars_scale_prior=10.0f
};

int main(int argc, char **argv) {
    DEBUG = (argc > 1) && (strcmp(argv[1], "--debug") == 0) ? 1 : 0;
    srand(0);

    const unsigned DATA_MEM_SIZE = N * sizeof(DTYPE);
    unsigned int zs[N], zsTrue[N];
    unsigned long runtime = 0;
    DTYPE weights[K], means[K], vars[K], *dataManaged = (DTYPE *) malloc(DATA_MEM_SIZE);;
    struct gmm_gibbs_state *gibbs_state;
    struct gmm_params params = {.weights=weights, .means=means, .vars=vars, .zs=zs};

    randomInit(dataManaged, zsTrue, N, K);
    rand_init_gmm_params(&params, N, K, PRIOR);
    gibbs_state = alloc_gmm_gibbs_state(N, K, dataManaged, PRIOR, &params);
    printParams(&params, dataManaged, N, K);

    begin_roi();
    gibbs(gibbs_state, ITERS);
    runtime = end_roi();
    printf("%d,%d,%lu\n", N, K, runtime);

//    DTYPE acc = accuracy(&params, zsTrue, N);

    free_gmm_gibbs_state(gibbs_state);
    printParams(&params, dataManaged, N, K);


    return 0;
}

void printParams(struct gmm_params *params, DTYPE *data, size_t n, size_t k) {
    printf("%lu\n", k);
    for (int i = 0; i < k; i++)
        printf("%f %f %f\n", params->weights[i], params->means[i], params->vars[i]);
    for (int i = 0; i < n; i++)
        printf("%.2f ", data[i]);
    putchar('\n');
    for (int i = 0; i < n; i++)
        printf("%u ", params->zs[i]);
    putchar('\n');
}

void randomInit(DTYPE *data, unsigned *zs, const int n, const int k) {
    std::default_random_engine generator;
    std::vector <std::vector<DTYPE>> samples;
    const int N_SAMPLES = 1024;
    int means[k], stds[k];

    // Generate arbitrary means and standard deviations
    for (int i = 0; i < k; i++) {
        std::normal_distribution <DTYPE> m(0, sqrt(n));
        std::normal_distribution <DTYPE> s(0, 4);
        means[i] = m(generator);
        stds[i] = abs(s(generator));
        printf("means[%d]/vars[%d] = %d/%d\n", i, i, means[i], stds[i] * stds[i]);
    }

    // Generate distributions, sample N_SAMPLES points, push to dist set
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

DTYPE accuracy(struct gmm_params *params, unsigned *zs, size_t n) {
    DTYPE err = 0.0;
    for (int i = 0; i < n; i++)
        if (params->zs[i] != zs[i])
            err += 1.0;
    return 1 - (err / n);
}
