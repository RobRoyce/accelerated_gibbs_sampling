#include "gmm.h"

struct gmm_sufficient_statistic {
    // `ns[i] = m` means m data points are assigned to the i-th component
    unsigned int *ns;

    // sum of data points in each component
    DTYPE *comp_sums;

    // sum of the squares of data points in each component
    DTYPE *comp_sqsums;
};

struct gmm_gibbs_state {
    // number of data points
    size_t n;

    // number of mixture components
    size_t k;

    DTYPE *data;

    struct gmm_prior prior;
    struct gmm_params *params;
    struct gmm_sufficient_statistic *ss;
};

struct gmm_gibbs_state *
alloc_gmm_gibbs_state(size_t n, size_t k, DTYPE *data, struct gmm_prior prior,
                      struct gmm_params *params) {
    struct gmm_gibbs_state *s = abortMalloc(sizeof(struct gmm_gibbs_state));
    s->n = n;
    s->k = k;
    s->data = data;
    s->prior = prior;
    s->params = params;
    s->ss = (struct gmm_sufficient_statistic *)
            abortMalloc(sizeof(struct gmm_sufficient_statistic));
    s->ss->ns = (unsigned *) abortCalloc(k, sizeof(unsigned int));
    s->ss->comp_sums = (DTYPE *) abortCalloc(k, sizeof(DTYPE));
    s->ss->comp_sqsums = (DTYPE *) abortCalloc(k, sizeof(DTYPE));
    return s;
}

void free_gmm_gibbs_state(struct gmm_gibbs_state *state) {
    free(state->ss->ns);
    free(state->ss->comp_sums);
    free(state->ss->comp_sqsums);
    free(state->ss);
    free(state);
}

void clear_sufficient_statistic(struct gmm_gibbs_state *state) {
    memset(state->ss->ns, 0, state->k * sizeof(unsigned int));
    memset(state->ss->comp_sums, 0, state->k * sizeof(DTYPE));
    memset(state->ss->comp_sqsums, 0, state->k * sizeof(DTYPE));
}

void update_sufficient_statistic(struct gmm_gibbs_state *state) {
    // XXX XXX this is the function that needs to be accelerated.
    clear_sufficient_statistic(state);
    for (size_t i = 0; i < state->n; i++) {
        DTYPE x = state->data[i];
        unsigned int z = state->params->zs[i];
        state->ss->ns[z]++;
        state->ss->comp_sums[z] += x;
        state->ss->comp_sqsums[z] += x * x;
    }
}

void update_ws(struct gmm_gibbs_state *state) {
    DTYPE dirichlet_param[state->k];
    vecAddUd(dirichlet_param, state->ss->ns, state->params->weights, state->k);
    dirichlet(state->params->weights, dirichlet_param, state->k);
}

void update_means(struct gmm_gibbs_state *state) {
    DTYPE k = 1 / state->prior.means_var_prior,
            zeta = state->prior.means_mean_prior, mean, var;
    for (int j = 0; j < state->k; j++) {
        DTYPE sum_xs = state->ss->comp_sums[j], ns = state->ss->ns[j],
                sigma2 = state->params->vars[j];
        mean = (k * zeta + sum_xs / sigma2) / (ns / sigma2 + k);
        var = 1 / (ns / sigma2 + k);
        state->params->means[j] = gaussian(mean, var);
    }
}

void update_vars(struct gmm_gibbs_state *state) {
    DTYPE alpha = state->prior.vars_shape_prior,
            beta = state->prior.vars_scale_prior, shape, scale;
    for (int j = 0; j < state->k; j++) {
        DTYPE sum_xs = state->ss->comp_sums[j],
                sqsum_xs = state->ss->comp_sqsums[j],
                mu = state->params->means[j], ns = state->ss->ns[j];
        shape = alpha + ns / 2;
        scale = beta + sqsum_xs / 2 - mu * sum_xs + ns * mu * mu / 2;
        state->params->vars[j] = inverse_gamma(shape, scale);
    }
}

void update_zs(struct gmm_gibbs_state *state) {
    DTYPE weights[state->k], mu, sigma2;
    for (int i = 0; i < state->n; i++) {
        DTYPE x = state->data[i];
        for (int j = 0; j < state->k; j++) {
            mu = state->params->means[j];
            sigma2 = state->params->vars[j];
            weights[j] = gaussian_pdf(x, mu, sigma2);
        }
        normalize(weights, state->k);
        state->params->zs[i] = categorical(weights, state->k);
    }
}

void gibbs(struct gmm_gibbs_state *state, size_t iters) {
    while (iters--) {
        update_sufficient_statistic(state);
        update_ws(state);
        update_means(state);
        update_vars(state);
        update_zs(state);
    }
}
