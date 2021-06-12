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

void printParams(struct gmm_params *params, DTYPE *data, size_t n, size_t k);

void randomInit(DTYPE *data, unsigned *zs, const int n, const int k);

void verify(struct gmm_params *params, unsigned *zs, size_t n);

const struct gmm_prior PRIOR = {
        .dirichlet_prior=5.0f,
        .means_mean_prior=0.0f,
        .means_var_prior=100.0f,
        .vars_shape_prior=2.0f,
        .vars_scale_prior=10.0f
};

int main(int argc, char **argv) {
    DEBUG = (argc > 1) && (strcmp(argv[1], "--debug") == 0) ? 1 : 0;
    srand(42);

    const unsigned DATA_MEM_SIZE = N * sizeof(DTYPE);
    unsigned int zs[N];
    DTYPE weights[K], means[K], vars[K], *dataManaged = malloc(DATA_MEM_SIZE);;
    struct gmm_gibbs_state *gibbs_state;
    struct gmm_params params = {.weights=weights, .means=means, .vars=vars, .zs=zs};

    randomInit(dataManaged, zs, N, K);
    rand_init_gmm_params(&params, N, K, PRIOR);
    gibbs_state = alloc_gmm_gibbs_state(N, K, dataManaged, PRIOR, &params);
//    printParams(&params, dataManaged, N, K);

    begin_roi();
    gibbs(gibbs_state, ITERS);
    end_roi();

    free_gmm_gibbs_state(gibbs_state);
//    printParams(&params, dataManaged, N, K);

    return 0;
}

void printParams(struct gmm_params *params, DTYPE *data, size_t n, size_t k) {
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
    unsigned cat = 0, mod = 0;;
    DTYPE min = 0;
    for (int i = 0; i < n; i++) {
        if (i < n / k) {
            min = 50;
            mod = 10;
            cat = 0;
        } else if (i < 2 * n / k) {
            min = 12;
            mod = 4;
            cat = 1;
        } else if (i < 3 * n / k) {
            min = -20;
            mod = 3;
            cat = 2;
        } else {
            min = -90;
            mod = 3;
            cat = 3;
        }
        data[i] = min + (DTYPE) (rand() % mod);
        zs[i] = cat;
    }
}

void verify(struct gmm_params *params, unsigned *zs, size_t n) {
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
