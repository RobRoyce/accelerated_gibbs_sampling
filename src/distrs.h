#pragma once
#include "utils.h"

DTYPE uniform(DTYPE a, DTYPE b);
DTYPE uniform_pdf(DTYPE x, DTYPE a, DTYPE b);
DTYPE uniform_cdf(DTYPE x, DTYPE a, DTYPE b);

int categorical(DTYPE *param, size_t n);
DTYPE categorical_pdf(int x, DTYPE *param, size_t n);
DTYPE categorical_cdf(int x, DTYPE *param, size_t n);

DTYPE gaussian(DTYPE mean, DTYPE var);
DTYPE gaussian_pdf(DTYPE x, DTYPE mean, DTYPE var);
DTYPE gaussian_cdf(DTYPE x, DTYPE mean, DTYPE var);

DTYPE gamma(DTYPE shape, DTYPE rate);
DTYPE gamma_pdf(DTYPE x, DTYPE shape, DTYPE rate);
DTYPE gamma_cdf(DTYPE x, DTYPE shape, DTYPE rate);

DTYPE inverse_gamma(DTYPE shape, DTYPE scale);
DTYPE inverse_gamma_pdf(DTYPE x, DTYPE shape, DTYPE scale);
DTYPE inverse_gamma_cdf(DTYPE x, DTYPE shape, DTYPE scale);

// sample from an `n`-dimensional dirichlet in `dst`
void dirichlet(DTYPE *dst, DTYPE *param, size_t n);
DTYPE dirichlet_pdf(DTYPE *x, DTYPE *param, size_t n);