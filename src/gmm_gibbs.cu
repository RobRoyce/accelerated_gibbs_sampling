#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <string.h>
#include "gmm.h"
#include "distrs.h"

#ifndef NSAMPLES
    #define NSAMPLES (1024)
#endif
#ifndef KCLASSES
    #define KCLASSES (16)
#endif

// cuRAND state array for uniform distributions
__device__ curandState curandStates[NSAMPLES];

__global__ void setup_kernel() {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(1234, id, 0, &(curandStates[id]));
}

void allocGmmGibbsState(struct GmmGibbsState **s, size_t n, size_t k, DTYPE *data,
                        struct GMMPrior prior, struct GMMParams *params) {
    gpuErrchk(cudaMallocManaged(s, sizeof(struct GmmGibbsState)));

    struct GmmGibbsState *state = *s;
    state->n = n;
    state->k = k;

    state->data = data;
    state->prior = prior;
    state->params = params;

    gpuErrchk(cudaMallocManaged(&(state->ss), sizeof(struct GmmSufficientStatistic)));
    gpuErrchk(cudaMallocManaged(&(state->ss->ns), k * sizeof(unsigned int)));
    gpuErrchk(cudaMallocManaged(&(state->ss->compSums), k * sizeof(DTYPE)));
    gpuErrchk(cudaMallocManaged(&(state->ss->compSquaredSums), k * sizeof(DTYPE)));

    // cudaMemset(state->ss, 0, sizeof(struct GmmSufficientStatistic));
    gpuErrchk(cudaMemset(state->ss->ns, 0, k * sizeof(unsigned int)));
    gpuErrchk(cudaMemset(state->ss->compSums, 0, k * sizeof(DTYPE)));
    gpuErrchk(cudaMemset(state->ss->compSquaredSums, 0, k * sizeof(DTYPE)));
}

void freeGmmGibbsState(struct GmmGibbsState *state) {
    gpuErrchk(cudaFree(state->ss->ns));
    gpuErrchk(cudaFree(state->ss->compSums));
    gpuErrchk(cudaFree(state->ss->compSquaredSums));
    gpuErrchk(cudaFree(state->ss));
    gpuErrchk(cudaFree(state));
}

__global__ void clearSufficientStatistic(struct GmmGibbsState *state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    state->ss->ns[i] = 0;
    state->ss->compSums[i] = 0;
    state->ss->compSquaredSums[i] = 0;
}

__global__ void updateSufficientStatistic(struct GmmGibbsState *state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    DTYPE x = state->data[i];
    unsigned int z = state->params->zs[i];

    atomicAdd(&(state->ss->ns[z]), 1);
    atomicAdd(&(state->ss->compSums[z]), x);
    atomicAdd(&(state->ss->compSquaredSums[z]), x * x);
}

void updateWeights(struct GmmGibbsState *state) {

    // DTYPE dirichletParam[state->k];
    DTYPE *dirichletParam;
    gpuErrchk(cudaMallocManaged(&dirichletParam, state->k * sizeof(DTYPE)));

    vecAddUd(dirichletParam, state->ss->ns, state->params->weights, state->k);
    dirichlet(state->params->weights, dirichletParam, state->k);

    gpuErrchk(cudaFree(dirichletParam));

}

__global__ void updateWeightsCuda(struct GmmGibbsState *state) {

    __shared__ DTYPE dirichletParam[KCLASSES]; // NOTE: shared memory only ~40kb. May be insufficient for large k

    vecAddUd(dirichletParam, state->ss->ns, state->params->weights, state->k);
    dirichlet(state->params->weights, dirichletParam, state->k);
}

__global__ void updateMeans(struct GmmGibbsState *state) {
    DTYPE k = 1 / state->prior.meansVarPrior,
            zeta = state->prior.meansMeanPrior,
            mean,
            var;

    int j = threadIdx.x + blockIdx.x * blockDim.x;

    DTYPE sum_xs = state->ss->compSums[j],
            ns = state->ss->ns[j],
            sigma2 = state->params->vars[j];

    mean = (k * zeta + sum_xs / sigma2) / (ns / sigma2 + k);
    var = 1 / (ns / sigma2 + k);

    state->params->means[j] = gaussian(mean, var);
}

__global__ void updateVars(struct GmmGibbsState *state) {
    DTYPE alpha = state->prior.varsShapePrior,
            beta = state->prior.varsScalePrior, shape, scale;

    int j = threadIdx.x + blockIdx.x * blockDim.x;

    DTYPE sum_xs = state->ss->compSums[j],
            sqsum_xs = state->ss->compSquaredSums[j],
            mu = state->params->means[j], ns = state->ss->ns[j];
    shape = alpha + ns / 2;
    scale = beta + sqsum_xs / 2 - mu * sum_xs + ns * mu * mu / 2;
    state->params->vars[j] = inverse_gamma(shape, scale);
}

__global__ void updateZs(struct GmmGibbsState *state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    DTYPE weights[KCLASSES];
    DTYPE mu, sigma2;

    DTYPE x = state->data[i];

    for (int j = 0; j < state->k; j++) {
        mu = state->params->means[j];
        sigma2 = state->params->vars[j];
        weights[j] = gaussian_pdf(x, mu, sigma2);
    }

    normalize(weights, state->k);
    state->params->zs[i] = categorical(weights, state->k);
}

void gibbs(struct GmmGibbsState *gibbsState, size_t iters) {
    dim3 kThreads(gibbsState->k, 1, 1);
    dim3 kBlocks(1, 1, 1);

    int thds = gibbsState->n < 1024 ? gibbsState->n : 1024;
    dim3 nThreads(thds, 1, 1);

    int blks = thds == 1024 ? gibbsState->n / nThreads.x : 1;
    dim3 nBlocks(blks, 1, 1);

    // Initialize CUDA random states
    setup_kernel<<<nThreads, nBlocks>>>();
    gpuErrchk(cudaDeviceSynchronize());

    while (iters--) {
        clearSufficientStatistic<<<kBlocks, kThreads>>>(gibbsState);
        updateSufficientStatistic<<<nBlocks, nThreads>>>(gibbsState);
        updateWeightsCuda<<<kBlocks, kThreads>>>(gibbsState);
        updateMeans<<<kBlocks, kThreads>>>(gibbsState);
        updateVars<<<kBlocks, kThreads>>>(gibbsState);
        updateZs<<<nBlocks, nThreads>>>(gibbsState);
    }

    gpuErrchk(cudaDeviceSynchronize());
}
