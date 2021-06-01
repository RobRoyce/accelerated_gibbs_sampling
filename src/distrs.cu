#include <stdlib.h>
#include <float.h>

#include <math.h>

#ifndef M_E
# define M_E		2.7182818284590452354	/* e */
#endif
#ifndef M_PI
# define M_PI		3.14159265358979323846	/* pi */
#endif

#include "utils.h"
#include "distrs.h"


double uniform(double a, double b)
{
    return a + rand() * (b - a) / RAND_MAX;
}

double uniform_pdf(double x, double a, double b)
{
    return x < b && x > a ? 1/(b - a) : 0;
}

double uniform_cdf(double x, double a, double b)
{
    return x < a ? 0 : x < b ? (x - a)/(b - a) : 1;
}

int categorical(double *param, size_t n)
{
    double u = uniform(0, 1), sum=0;
    int i = 0;
    while(u > sum)
        sum += param[i++];
    return i - 1;
}

double categorical_pdf(int x, double *param, size_t n)
{
    if(x < 0 || x >= n)
        return 0;
    return param[x];
}

double categorical_cdf(int x, double *param, size_t n)
{
    if(x < 0 || x >= n)
        return 0;
    double res = param[0];
    while(x)
        res += param[x--];
    return res;
}

double gaussian(double mean, double var)
{
    double u=uniform(0, 1), v=uniform(0, 1);
    return mean + sqrt(-2 * log(u) * var) * cos(2*M_PI * v);
}

double gaussian_pdf(double x, double mean, double var)
{
    return exp(-square(x - mean)/2/var) / sqrt(2*M_PI * var);
}

double gaussian_cdf(double x, double mean, double var)
{
    return (1 + erf((x - mean) / sqrt(2 * var))) / 2;
}

//Wikipedia algorithm
double gamma(double shape, double rate)
{
    int n = floor(shape);
    double delta = shape - n, exp_part = 0,  xi, eta;
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

double gamma_pdf(double x, double shape, double rate)
{
    return x <= 0 ? 0 : pow(rate*x, shape) / tgamma(shape) * exp(-rate*x)/x;
}

double gamma_cdf(double x, double shape, double rate)
{
    return x <= 0 ? 0 : ligamma(shape, rate * x) / tgamma(shape);
}

double inverse_gamma(double shape, double scale)
{
    return 1 / gamma(shape, scale);
}

double inverse_gamma_pdf(double x, double shape, double scale)
{
    return x <= 0 ? 0 : pow(scale/x, shape) / tgamma(shape) * exp(-scale/x)/x;
}

double inverse_gamma_cdf(double x, double shape, double scale)
{
    return x <= 0 ? 0 : 1 - ligamma(shape, scale / x) / tgamma(shape);
}

void dirichlet(double *dst, double *param, size_t n)
{
    for(int i=0; i < n; i++)
        dst[i] = gamma(param[i], 1);
    normalize(dst, n);
}

double dirichlet_pdf(double *x, double *param, size_t n)
{
    double sum=0, unnorm_pdf=1;
    for(int i=0; i < n; sum += x[i], unnorm_pdf *= pow(x[i], param[i]-1), i++);
    return sum == 1.0 ? unnorm_pdf / beta(x, n) : 0.0;
}
