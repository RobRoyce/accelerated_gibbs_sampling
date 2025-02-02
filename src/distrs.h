#pragma once
#include "utils.h"

__device__ DTYPE uniform_cuda(DTYPE a, DTYPE b, DTYPE *dest, size_t n);
//int categorical_cuda(DTYPE *param, size_t n);
//DTYPE gaussian_cuda(DTYPE mean, DTYPE var);
//DTYPE gamma_cuda(DTYPE shape, DTYPE rate);
//DTYPE inverse_gamma_cuda(DTYPE shape, DTYPE scale);

__host__ __device__ DTYPE uniform(DTYPE a, DTYPE b);
DTYPE uniform_pdf(DTYPE x, DTYPE a, DTYPE b);
DTYPE uniform_cdf(DTYPE x, DTYPE a, DTYPE b);

__host__ __device__ int categorical(DTYPE *param, size_t n);
DTYPE categorical_pdf(int x, DTYPE *param, size_t n);
DTYPE categorical_cdf(int x, DTYPE *param, size_t n);

__host__ __device__ DTYPE gaussian(DTYPE mean, DTYPE var);
__host__ __device__ DTYPE gaussian_pdf(DTYPE x, DTYPE mean, DTYPE var);
DTYPE gaussian_cdf(DTYPE x, DTYPE mean, DTYPE var);

__host__ __device__ DTYPE gamma(DTYPE shape, DTYPE rate);
DTYPE gamma_pdf(DTYPE x, DTYPE shape, DTYPE rate);
DTYPE gamma_cdf(DTYPE x, DTYPE shape, DTYPE rate);

__host__ __device__ DTYPE inverse_gamma(DTYPE shape, DTYPE scale);
DTYPE inverse_gamma_pdf(DTYPE x, DTYPE shape, DTYPE scale);
DTYPE inverse_gamma_cdf(DTYPE x, DTYPE shape, DTYPE scale);

// sample from an `n`-dimensional dirichlet in `dst`
__host__ __device__ void dirichlet(DTYPE *dst, DTYPE *param, size_t n);
DTYPE dirichlet_pdf(DTYPE *x, DTYPE *param, size_t n);
