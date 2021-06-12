#pragma once

#include <math.h>

typedef float DTYPE;

// add `u` and `v` of length `k` and store the result in `dst`
void vecAddDd(DTYPE *dst, DTYPE *u, DTYPE *v, size_t k);
__host__ __device__ void vecAddUd(DTYPE *dst, unsigned int *u, DTYPE *v, size_t k);

__host__ __device__ void normalize(DTYPE *v, size_t n);

DTYPE square(DTYPE x);

DTYPE ligamma(DTYPE s, DTYPE x);

DTYPE uigamma(DTYPE s, DTYPE x);

DTYPE beta(DTYPE *x, size_t n);

DTYPE zScore(DTYPE m1, DTYPE m2, DTYPE var1, DTYPE var2);

DTYPE conflateVar(DTYPE var1, DTYPE var2);

DTYPE conflateMean(DTYPE m1, DTYPE m2, DTYPE var1, DTYPE var2);

// try to malloc/calloc and abort if unsuccessful
void *abortCalloc(size_t nmemb, size_t size);
void *abortMalloc(size_t size);
