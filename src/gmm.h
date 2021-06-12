#pragma once

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "gpu.h"
#include "distrs.h"


struct GMMPrior {
    // symmetric Dirichlet prior on component weights
    DTYPE dirichletPrior;

    // the mean parameter of the Gaussian prior on component means
    DTYPE meansMeanPrior;

    // the variance parameter of the Gaussian prior on component means
    DTYPE meansVarPrior;

    // the shape parameter of the inverse-gamma prior on component variance
    DTYPE varsShapePrior;

    // the scale parameter of the inverse-gamma prior on component variance
    DTYPE varsScalePrior;
};

struct GMMParams {
    DTYPE *weights;
    DTYPE *means;
    DTYPE *vars;

    // the latent allocation variables; `z[i] = j` means that the i-th data
    // point is sampled from the j-th component
    unsigned int *zs;
};

struct GmmSufficientStatistic {
    // `ns[i] = m` means m data points are assigned to the i-th component
    unsigned int *ns;

    // sum of data points in each component
    DTYPE *compSums;

    // sum of the squares of data points in each component
    DTYPE *compSquaredSums;
};

struct GmmGibbsState {
    // number of data points
    size_t n;

    // number of mixture components
    size_t k;

    DTYPE *data;

    struct GMMPrior prior;
    struct GMMParams *params;
    struct GmmSufficientStatistic *ss;
};

void randInitGmmParams(struct GMMParams *params, size_t n, size_t k,
                       struct GMMPrior prior);

void allocGmmGibbsState(struct GmmGibbsState **s, size_t n, size_t k, size_t m, DTYPE *data, struct GMMPrior prior);

void freeGmmGibbsState(struct GmmGibbsState *state, size_t m);

void gibbs(struct GmmGibbsState *states, int num_states, size_t iters);
