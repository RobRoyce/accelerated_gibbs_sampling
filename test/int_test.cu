#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../src/gmm.h"
#include "../src/utils.h"
#include "../src/gpu.h"

int DEBUG = 1;

DTYPE DATA[] = {11.26, 28.93, 30.52, 30.09, 29.46, 10.03, 11.24, 11.55,
                 30.4, -18.44, 10.91, 11.89, -20.64, 30.59, 14.84, 13.54, 7.25, 12.83,
                 11.86, 29.95, 29.47, -18.16, -19.28, -18.87, 9.95, 28.24, 9.43, 7.38,
                 29.46, 30.73, 7.75, 28.29, -21.99, -20.0, -20.86, 15.5, -18.62, 13.11,
                 28.66, 28.18, -18.78, -20.48, 9.18, -20.12, 10.2, 30.26, -14.94, 5.45,
                 31.1, 30.01, 10.52, 30.48, -20.37, -19.3, -21.92, -18.31, -18.9, -20.03,
                 29.32, -17.53, 10.61, 6.38, -20.72, 10.29, 11.21, -18.98, 8.57, 10.47,
                 -22.4, 6.58, 29.8, -17.43, 7.8, 9.72, -21.53, 11.76, 29.72, 29.31, 6.82,
                 15.51, 10.69, 29.56, 8.84, 30.93, 28.75, 10.72, 9.21, 8.57, 11.92, -23.96,
                 -19.78, -17.2, 11.79, 29.95, 7.29, 6.57, -17.99, 13.29, -22.53, -20.0};


const unsigned int ZS[] = {1, 2, 2, 2, 2, 1, 1, 1, 2, 0, 1, 1, 0, 2, 1, 1, 1,
                           1, 1, 2, 2, 0, 0, 0, 1, 2, 1, 1, 2, 2, 1, 2, 0, 0, 0, 1, 0, 1, 2, 2, 0, 0,
                           1, 0, 1, 2, 0, 1, 2, 2, 1, 2, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 1, 1, 0, 1,
                           1, 0, 1, 2, 0, 1, 1, 0, 1, 2, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 0, 0, 0,
                           1, 2, 1, 1, 0, 1, 0, 0};

const int N = sizeof(DATA) / sizeof(DTYPE);
const int K = 3;
const int ITERS = 500;

const struct gmm_prior PRIOR = {.dirichlet_prior=5.0,
        .means_mean_prior=0.0,
        .means_var_prior=100.0,
        .vars_shape_prior=2.0,
        .vars_scale_prior=10.0};

void verify(struct gmm_params *params) {
    int c1, c2, c3, err = 0;
    c1 = params->zs[0];
    c2 = params->zs[1];
    c3 = params->zs[N - 1];

    for (int i = 0; i < N; i++) {
        if ((ZS[i] == 1 && params->zs[i] != c1) || (ZS[i] == 2 && params->zs[i] != c2) ||
            (ZS[i] == 0 && params->zs[i] != c3)) {
            err = 1;
        }
    }
    if (err != 0)
        printf("int_test ---------------------------------------- FAILED! \n");
    else
        printf("int_test ---------------------------------------- SUCCESS! \n");
}

void print_params(struct gmm_params *params) {
    printf("%d\n", K);
    for (int i = 0; i < K; i++)
        printf("%f %f %f\n", params->weights[i], params->means[i],
               params->vars[i]);
    for (int i = 0; i < N; i++)
        printf("%.2f ", DATA[i]);
    putchar('\n');
    for (int i = 0; i < N; i++)
        printf("%u ", params->zs[i]);
    putchar('\n');
}

int main(int argc, char **argv) {
    DEBUG = (argc > 1) && (strcmp(argv[1], "--debug") == 0) ? 1 : 0;

    struct gmm_gibbs_state *gibbs_state;
    struct gmm_params *params;
    DTYPE *data_managed;

    gpuErrchk(cudaMallocManaged(&params, sizeof(struct gmm_params)));
    gpuErrchk(cudaMallocManaged(&(params->weights), K * sizeof(DTYPE)));
    gpuErrchk(cudaMallocManaged(&(params->means), K * sizeof(DTYPE)));
    gpuErrchk(cudaMallocManaged(&(params->vars), K * sizeof(DTYPE)));
    gpuErrchk(cudaMallocManaged(&(params->zs), K * sizeof(unsigned int)));
    gpuErrchk(cudaMallocManaged(&data_managed, sizeof(DATA)));
    gpuErrchk(cudaMemcpy(data_managed, DATA, sizeof(DATA), cudaMemcpyDefault));

    srand(time(NULL));
    rand_init_gmm_params(params, N, K, PRIOR);

    gibbs_state = alloc_gmm_gibbs_state(N, K, data_managed, PRIOR, params);
    gibbs(gibbs_state, ITERS);

    free_gmm_gibbs_state(gibbs_state);
//    print_params(params);
    verify(params);
    return 0;
}
