#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "gmm.h"
#include "utils.h"
#include "distrs.h"
#include "cuda.h"

__device__ struct gmm_gibbs_state *d_state;

struct gmm_sufficient_statistic {
    // `ns[i] = m` means m data points are assigned to the i-th component
    unsigned int *ns; 

    // sum of data points in each component
    double *comp_sums;

    // sum of the squares of data points in each component
    double *comp_sqsums;
};

struct gmm_gibbs_state {
    // number of data points
    size_t n;

    // number of mixture components
    size_t k; 

    double *data;

    struct gmm_prior prior;
    struct gmm_params *params;
    struct gmm_sufficient_statistic *ss;
};

struct gmm_gibbs_state *
alloc_gmm_gibbs_state(size_t n, size_t k, double *data, struct gmm_prior prior,
                      struct gmm_params *params)
{
    struct gmm_gibbs_state *s;
    cudaMallocManaged(&s, sizeof(struct gmm_gibbs_state));
    
    s->n = n;
    s->k = k;
    s->data = data;
    s->prior = prior;
    s->params = params;

    cudaMallocManaged(&(s->ss), sizeof(struct gmm_sufficient_statistic));
    cudaMallocManaged(&(s->ss->ns), k*sizeof(unsigned int));
    cudaMallocManaged(&(s->ss->comp_sums), k*sizeof(double));
    cudaMallocManaged(&(s->ss->comp_sqsums), k*sizeof(double));

    // cudaMemset(s->ss, 0, sizeof(struct gmm_sufficient_statistic));
    cudaMemset(s->ss->ns, 0, k*sizeof(unsigned int));
    cudaMemset(s->ss->comp_sums, 0, k*sizeof(double));
    cudaMemset(s->ss->comp_sqsums, 0, k*sizeof(double));

    return s;
}

void free_gmm_gibbs_state(struct gmm_gibbs_state *state)
{
    cudaFree(state->ss->ns);
    cudaFree(state->ss->comp_sums);
    cudaFree(state->ss->comp_sqsums);
    cudaFree(state->ss);
    cudaFree(state);
}

void clear_sufficient_statistic(struct gmm_gibbs_state *state)
{
    cudaMemset(state->ss->ns, 0, state->k * sizeof(unsigned int));
    cudaMemset(state->ss->comp_sums, 0, state->k * sizeof(double));
    cudaMemset(state->ss->comp_sqsums, 0, state->k * sizeof(double));
}

__global__ void update_sufficient_statistic_cuda(struct gmm_gibbs_state *state)
{
    // printf("Ok...\n");
    int i = threadIdx.x;
    double x = state->data[i];
    unsigned int z = state->params->zs[i];

    // state->ss->ns[z]++;
    // state->ss->comp_sums[z] += x;
    // state->ss->comp_sqsums[z] += x*x;
    atomicAdd(&(state->ss->ns[z]), 1);
    atomicAdd(&(state->ss->comp_sums[z]), x);
    atomicAdd(&(state->ss->comp_sqsums[z]), x*x);
}

void update_sufficient_statistic(struct gmm_gibbs_state *state)
{
    // XXX XXX this is the function that needs to be accelerated.
    clear_sufficient_statistic(state);
    for(size_t i=0; i < state->n; i++) {
        double x = state->data[i];
        unsigned int z = state->params->zs[i];
        state->ss->ns[z]++;
        state->ss->comp_sums[z] += x;
        state->ss->comp_sqsums[z] += x*x;
    }
}

void update_ws(struct gmm_gibbs_state *state)
{
    // double dirichlet_param[state->k];
    double *dirichlet_param;
    cudaMallocManaged(&dirichlet_param, state->k * sizeof(double));

    vec_add_ud(dirichlet_param, state->ss->ns, state->params->weights, state->k);
    dirichlet(state->params->weights, dirichlet_param, state->k);

    cudaFree(dirichlet_param);
}

void update_means(struct gmm_gibbs_state *state)
{
    double k = 1/state->prior.means_var_prior,
           zeta = state->prior.means_mean_prior, mean, var;
    for(int j=0; j < state->k; j++) {
        double sum_xs = state->ss->comp_sums[j], ns = state->ss->ns[j],
               sigma2 = state->params->vars[j];
               mean = (k * zeta + sum_xs / sigma2) / (ns / sigma2 + k);
               var = 1/(ns / sigma2 + k);
        state->params->means[j] = gaussian(mean, var);
    }
}

void update_vars(struct gmm_gibbs_state *state)
{
    double alpha = state->prior.vars_shape_prior,
           beta = state->prior.vars_scale_prior, shape, scale;
    for(int j=0; j < state->k; j++) {
        double sum_xs = state->ss->comp_sums[j],
               sqsum_xs = state->ss->comp_sqsums[j],
               mu = state->params->means[j], ns = state->ss->ns[j];
               shape = alpha + ns/2;
               scale = beta + sqsum_xs/2 - mu*sum_xs + ns * mu*mu/2; 
        state->params->vars[j] = inverse_gamma(shape, scale);
    }
}

void update_zs(struct gmm_gibbs_state *state)
{
    double weights[state->k], mu, sigma2;
    for(int i=0; i < state->n; i++) {
        double x = state->data[i];
        for(int j=0; j < state->k; j++) {
            mu = state->params->means[j];
            sigma2 = state->params->vars[j];
            weights[j] = gaussian_pdf(x, mu, sigma2);
        }
        normalize(weights, state->k);
        state->params->zs[i] = categorical(weights, state->k);
    }
}

void gibbs(struct gmm_gibbs_state *state, size_t iters)
{
    int numBlocks = 1;
    dim3 threadsPerBlock(state->n);

    while(iters--) {

        clear_sufficient_statistic(state);
        update_sufficient_statistic_cuda<<<numBlocks, threadsPerBlock>>>(state);
        cudaDeviceSynchronize();

        update_ws(state);
        update_means(state);
        update_vars(state);
        update_zs(state);
    }
}
