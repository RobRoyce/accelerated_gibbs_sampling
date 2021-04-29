#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "gmm.h"
#include "utils.h"
#include "distrs.h"

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

void alloc_gmm_sufficient_statistic(struct gmm_sufficient_statistic *ss)
{
    ss->ns = (unsigned int *) calloc(ss->k, sizeof unsigned int);
    ss->comp_sums = (double *) calloc(ss->k, sizeof double);
    ss->comp_sqsums = (double *) calloc(ss->k, sizeof double);
    if(!ss->ns || !ss->comp_sums || !ss->comp_sqsums) {
        fputs(stderr, "insufficient memory");
        exit(1);
    }
}

void free_gmm_sufficient_statistic(struct gmm_sufficient_statistic *ss)
{
    free(ss-ns);
    free(ss->comp_sums);
    free(ss->comp_sqsums);
}

void alloc_gmm_gibbs_state(struct gmm_gibbs_state *state,
                           size_t n, size_t k, double *data,
                           struct gmm_prior prior,
                           struct gmm_params *params)
{
    state->n = n;
    state->k = k;
    state->data = data;
    state->prior = prior;
    state->params = params;
    alloc_gmm_sufficient_statistic(state->ss);
}

void free_gmm_gibbs_state(struct gmm_gibbs_state *state)
{
    free_gmm_sufficient_statistic(state->ss);
}

void clear_sufficient_statistic(struct gmm_gibbs_state *state)
{
    memset(state->ss->ns, 0, state->k * (sizeof unsigned int));
    memset(state->ss->comp_sums, 0, state->k * (sizeof double));
    memset(state->ss->comp_sqsums, 0, state->k * (sizeof double));
}

void update_sufficient_statistic(struct gmm_gibbs_state *state)
{
    // XXX XXX this is the function that needs to be accelerated.
    clear_sufficient_statistic(state);
    for(size_t i = 0; i < state->n; i++) {
        double x = data[i];
        unsigned int z = state->params->zs[i];
        state->ss->ns[z]++;
        state->ss->comp_sums[z] += x;
        state->ss->comp_sqsums[z] += x*x;
    }
}

void update_ws(struct gmm_gibbs_state *state)
{
    double dirichlet_param[state->prior.k];
    vec_add(dirichlet_param, state->ss->ns, state->params->weights);
    dirichlet(state->params->weights, dirichlet_param);
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
        state->params->means[j] = gaussian(mean, var)
    }
}

void update_vars(struct gmm_gibbs_state *state)
{
    double alpha = state->prior.vars_shape_prior,
           beta = state->prior.vars_shape_prior, shape, rate;
    for(int j=0; j < state->k; j++) {
        double sum_xs = state->ss->comp_sums[j],
               sqsum_xs = state->ss->comp_sqsums[j],
               mu = state->params->means[j], ns = state->ss->ns[j];
               shape = alpha + ns/2;
               rate = beta + sqsum_xs - 2*mu*sum_xs, + ns * mu*mu; 
        state->params->vars[j] = inverse_gamma(shape, rate);
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
            weights[j] = gaussian_pdf(mu, sigma2, x);
        }
        normalize(weights);
        state->params->zs[i] = categorical(weights);
    }
}

void gibbs(struct gmm_gibbs_state *state, size_t iters)
{
    while(iters--) {
        update_sufficient_statistics(state);
        update_ws(state);
        update_means(state);
        update_vars(state);
        update_zs(state);
    }
}
