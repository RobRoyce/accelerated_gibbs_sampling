#pragma once

#include <math.h>

typedef float DTYPE;

// add `u` and `v` of length `k` and store the result in `dst`
void vec_add_dd(DTYPE *dst, DTYPE *u, DTYPE *v, size_t k);
void vec_add_ud(DTYPE *dst, unsigned int *u, DTYPE *v, size_t k);

void normalize(DTYPE *v, size_t n);

DTYPE square(DTYPE x);

DTYPE ligamma(DTYPE s, DTYPE x);

extern DTYPE uigamma(DTYPE s, DTYPE x);

DTYPE beta(DTYPE *x, size_t n);

// try to malloc/calloc and abort if unsuccessful
void *abort_calloc(size_t nmemb, size_t size);
void *abort_malloc(size_t size);
