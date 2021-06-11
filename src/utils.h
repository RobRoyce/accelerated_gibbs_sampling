#pragma once

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

typedef float DTYPE;

// add `u` and `v` of length `k` and store the result in `dst`
void vecAddDd(DTYPE *dst, DTYPE *u, DTYPE *v, size_t k);

void vecAddUd(DTYPE *dst, unsigned int *u, DTYPE *v, size_t k);

void normalize(DTYPE *v, size_t n);

DTYPE square(DTYPE x);

DTYPE ligamma(DTYPE s, DTYPE x);

extern DTYPE uigamma(DTYPE s, DTYPE x);

DTYPE beta(DTYPE *x, size_t n);

// try to malloc/calloc and abort if unsuccessful
void *abortCalloc(size_t nmemb, size_t size);

void *abortMalloc(size_t size);