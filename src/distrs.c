#include <stdlib.h>
#include <float.h>
#include <math.h>

#ifndef M_E
# define M_E		2.7182818284590452354	/* e */
#endif
#ifndef M_PI
# define M_PI		3.14159265358979323846	/* pi */
#endif

#include "distrs.h"


DTYPE uniform(DTYPE a, DTYPE b)
{
    return a + rand() * (b - a) / RAND_MAX;
}

DTYPE uniform_pdf(DTYPE x, DTYPE a, DTYPE b)
{
    return x < b && x > a ? 1/(b - a) : 0;
}

DTYPE uniform_cdf(DTYPE x, DTYPE a, DTYPE b)
{
    return x < a ? 0 : x < b ? (x - a)/(b - a) : 1;
}

int categorical(DTYPE *param, size_t n)
{
    DTYPE u = uniform(0, 1), sum=0;
    int i = 0;
    while(u > sum)
        sum += param[i++];
    return i - 1;
}

DTYPE categorical_pdf(int x, DTYPE *param, size_t n)
{
    if(x < 0 || x >= n)
        return 0;
    return param[x];
}

DTYPE categorical_cdf(int x, DTYPE *param, size_t n)
{
    if(x < 0 || x >= n)
        return 0;
    DTYPE res = param[0];
    while(x)
        res += param[x--];
    return res;
}

DTYPE gaussian(DTYPE mean, DTYPE var)
{
    DTYPE u=uniform(0, 1), v=uniform(0, 1);
    return mean + sqrt(-2 * log(u) * var) * cos(2*M_PI * v);
}

DTYPE gaussian_pdf(DTYPE x, DTYPE mean, DTYPE var)
{
    return exp(-square(x - mean)/2/var) / sqrt(2*M_PI * var);
}

DTYPE gaussian_cdf(DTYPE x, DTYPE mean, DTYPE var)
{
    return (1 + erf((x - mean) / sqrt(2 * var))) / 2;
}

//Wikipedia algorithm
DTYPE gamma(DTYPE shape, DTYPE rate)
{
    int n = floor(shape);
    DTYPE delta = shape - n, exp_part = 0,  xi = 0, eta = 0;
    while(n--)
        exp_part += -log(uniform(0, 1));
    if(delta > DBL_EPSILON) do {
        if(uniform(0, 1) < M_E / (M_E + delta)) {
            xi = pow(uniform(0, 1), 1/delta);
            eta = uniform(0, 1) * pow(xi, delta - 1);
        } else {
            xi = 1 - log(uniform(0, 1));
            eta = uniform(0, 1) * exp(-xi);
        }
    } while(eta > pow(xi, delta -1) * exp(-xi));
    return (xi + exp_part) /rate;
}

DTYPE gamma_pdf(DTYPE x, DTYPE shape, DTYPE rate)
{
    return x <= 0 ? 0 : pow(rate*x, shape) / tgamma(shape) * exp(-rate*x)/x;
}

DTYPE gamma_cdf(DTYPE x, DTYPE shape, DTYPE rate)
{
    return x <= 0 ? 0 : ligamma(shape, rate * x) / tgamma(shape);
}

DTYPE inverse_gamma(DTYPE shape, DTYPE scale)
{
    return 1/gamma(shape, scale);
}

DTYPE inverse_gamma_pdf(DTYPE x, DTYPE shape, DTYPE scale)
{
    return x <= 0 ? 0 : pow(scale/x, shape) / tgamma(shape) * exp(-scale/x)/x;
}

DTYPE inverse_gamma_cdf(DTYPE x, DTYPE shape, DTYPE scale)
{
    return x <= 0 ? 0 : 1 - ligamma(shape, scale / x) / tgamma(shape);
}

void dirichlet(DTYPE *dst, DTYPE *param, size_t n)
{
    for(int i=0; i < n; i++)
        dst[i] = gamma(param[i], 1);
    normalize(dst, n);
}

DTYPE dirichlet_pdf(DTYPE *x, DTYPE *param, size_t n)
{
    DTYPE sum=0, unnorm_pdf=1;
    for(int i=0; i < n; sum += x[i], unnorm_pdf *= pow(x[i], param[i]-1), i++);
    return sum == 1.0 ? unnorm_pdf / beta(x, n) : 0.0;
}
