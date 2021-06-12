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
#ifndef MSAMPLERS
    #define MSAMPLERS (4)
#endif

// cuRAND state array for uniform distributions
__device__ curandState curandStates[MSAMPLERS];

__global__ void setup_kernel() {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(1234, id, 0, &(curandStates[id]));
}

inline void swap(DTYPE *data, int i, int j)
{
    DTYPE tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
}

void shuffle(DTYPE *data, int n)
{
    for(int i = 0; i < n; i++)
    {
        unsigned int swapIdx = (unsigned int)rand() % n;
        swap(data, i, swapIdx);
    }
}

void allocGmmGibbsState(struct GmmGibbsState **s, size_t n, size_t k, size_t m, DTYPE *data, struct GMMPrior prior) {

    gpuErrchk(cudaMallocManaged(s, m * sizeof(struct GmmGibbsState)));

    // Shuffle the dataset such that each independent sampler gets a decent representation of the problem
    shuffle(data, n);

    struct GmmGibbsState *state;
    for(int i = 0; i < m; i++)
    {    
        const unsigned CLASS_MEM_SIZE = k * sizeof(DTYPE),
                PARAM_MEM_SIZE = sizeof(struct GMMParams),
                ZS_MEM_SIZE = n * sizeof(unsigned);
        struct GMMParams *params = nullptr;

        gpuErrchk(cudaMallocManaged(&params, PARAM_MEM_SIZE));
        gpuErrchk(cudaMallocManaged(&(params->weights), CLASS_MEM_SIZE));
        gpuErrchk(cudaMallocManaged(&(params->means), CLASS_MEM_SIZE));
        gpuErrchk(cudaMallocManaged(&(params->vars), CLASS_MEM_SIZE));
        gpuErrchk(cudaMallocManaged(&(params->zs), ZS_MEM_SIZE));

        state = &(*s)[i];
        state->n = n/m;
        state->k = k;

        state->data = &data[state->n * i];
        state->prior = prior;

        randInitGmmParams(params, n, k, prior);

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
}

void freeGmmGibbsState(struct GmmGibbsState *state, size_t m) {
    for (int i=0; i < m; i++)
    {
        struct GmmGibbsState *s = &state[i];
        gpuErrchk(cudaFree(s->ss->ns));
        gpuErrchk(cudaFree(s->ss->compSums));
        gpuErrchk(cudaFree(s->ss->compSquaredSums));
        gpuErrchk(cudaFree(s->ss));
        gpuErrchk(cudaFree(s));
    }
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
    struct GmmGibbsState *state = &gibbsStates[i];

    while (iters--) {
        clearSufficientStatistic(state);
        updateSufficientStatistic(state);
        updateWeights(state);
        updateMeans(state);
        updateVars(state);
        updateZs(state);
    }

}

void gibbs(struct GmmGibbsState *gibbsStates, int num_states, size_t iters) {
    
    if(num_states < 32)
    {
        // Initialize CUDA random states
        setup_kernel<<<num_states, 1>>>();

        // Run independent Gibbs samplers
        gibbsCuda<<<num_states, 1>>>(gibbsStates, iters);
    }
    else
    {
        // Initialize CUDA random states
        setup_kernel<<<32, num_states/32>>>();

        // Run independent Gibbs samplers
        gibbsCuda<<<32, num_states/32>>>(gibbsStates, iters);
    }

    gpuErrchk(cudaDeviceSynchronize());
}
