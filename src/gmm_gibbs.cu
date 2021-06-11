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

__device__ void clearSufficientStatistic(struct GmmGibbsState *state) {
    for(int i = 0; i < state->k; i++)
    {
        state->ss->ns[i] = 0;
        state->ss->compSums[i] = 0;
        state->ss->compSquaredSums[i] = 0;
    }
}

__device__ void updateSufficientStatistic(struct GmmGibbsState *state) {
    for(size_t i=0; i < state->n; i++) {
        DTYPE x = state->data[i];
        unsigned int z = state->params->zs[i];
        state->ss->ns[z]++;
        state->ss->compSums[z] += x;
        state->ss->compSquaredSums[z] += x*x;
    }
}

__device__ void updateWeights(struct GmmGibbsState *state) {

    DTYPE dirichlet_param[KCLASSES];
    vecAddUd(dirichlet_param, state->ss->ns, state->params->weights, state->k);
    dirichlet(state->params->weights, dirichlet_param, state->k);

}

__device__ void updateMeans(struct GmmGibbsState *state) {
    DTYPE k = 1/state->prior.meansVarPrior,
           zeta = state->prior.meansMeanPrior, mean, var;
    for(int j=0; j < state->k; j++) {
        DTYPE sum_xs = state->ss->compSums[j], ns = state->ss->ns[j],
               sigma2 = state->params->vars[j];
               mean = (k * zeta + sum_xs / sigma2) / (ns / sigma2 + k);
               var = 1/(ns / sigma2 + k);
        state->params->means[j] = gaussian(mean, var);
    }
}

__device__ void updateVars(struct GmmGibbsState *state) {
    DTYPE alpha = state->prior.varsShapePrior,
           beta = state->prior.varsScalePrior, shape, scale;
    for(int j=0; j < state->k; j++) {
        DTYPE sum_xs = state->ss->compSums[j],
               sqsum_xs = state->ss->compSquaredSums[j],
               mu = state->params->means[j], ns = state->ss->ns[j];
               shape = alpha + ns/2;
               scale = beta + sqsum_xs/2 - mu*sum_xs + ns * mu*mu/2; 
        state->params->vars[j] = inverse_gamma(shape, scale);
    }
}

__device__ void updateZs(struct GmmGibbsState *state) {
    DTYPE weights[KCLASSES], mu, sigma2;
    for(int i=0; i < state->n; i++) {
        DTYPE x = state->data[i];
        for(int j=0; j < state->k; j++) {
            mu = state->params->means[j];
            sigma2 = state->params->vars[j];
            weights[j] = gaussian_pdf(x, mu, sigma2);
        }
        normalize(weights, state->k);
        state->params->zs[i] = categorical(weights, state->k);
    }
}

__global__ void gibbsCuda(struct GmmGibbsState *gibbsStates, size_t iters) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    while (iters--) {
        clearSufficientStatistic(&gibbsStates[i]);
        updateSufficientStatistic(&gibbsStates[i]);
        updateWeights(&gibbsStates[i]);
        updateMeans(&gibbsStates[i]);
        updateVars(&gibbsStates[i]);
        updateZs(&gibbsStates[i]);
    }


}

void gibbs(struct GmmGibbsState *gibbsStates, int num_states, size_t iters) {
    dim3 nThreads(1024, 1, 1);
    dim3 nBlocks(gibbsStates->n / nThreads.x, 1, 1);

    // Initialize CUDA random states
    setup_kernel<<<1024, (gibbsStates->n * num_states) / 1024>>>();

    // Run independent Gibbs samplers
    // gibbsCuda<<<32, num_states/32>>>(gibbsStates, iters);
    gibbsCuda<<<1, 1>>>(gibbsStates, iters);
    gpuErrchk(cudaDeviceSynchronize());
}
