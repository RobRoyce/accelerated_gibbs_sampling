#pragma once

#include <stddef.h>

struct gmm_prior {
    // symmetric Dirichlet prior on component weights
    double dirichlet_prior; 

    // the mean parameter of the Gaussian prior on component means
    double means_mean_prior;

    // the variance parameter of the Gaussian prior on component means
    double means_var_prior;

    // the shape parameter of the inverse-gamma prior on component variance
    double vars_shape_prior;

    // the scale parameter of the inverse-gamma prior on component variance
    double vars_scale_prior;
};

struct gmm_params {
    double *weights;
    double *means; 
    double *vars; 

    // the latent allocation variables; `z[i] = j` means that the i-th data
    // point is sampled from the j-th component
    unsigned int *zs;
};

struct gmm_gibbs_state;

void rand_init_gmm_params(struct gmm_params *params, size_t n, size_t k,
                          struct gmm_prior prior);

struct gmm_gibbs_state *alloc_gmm_gibbs_state(size_t n, size_t k, double *data,
                                              struct gmm_prior prior,
                                              struct gmm_params *params);

void free_gmm_gibbs_state(struct gmm_gibbs_state *state);

void gibbs(struct gmm_gibbs_state *state, size_t iters);
