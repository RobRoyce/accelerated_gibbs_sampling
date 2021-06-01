#pragma once

double uniform(double a, double b);
double uniform_pdf(double x, double a, double b);
double uniform_cdf(double x, double a, double b);

int categorical(double *param, size_t n);
double categorical_pdf(int x, double *param, size_t n);
double categorical_cdf(int x, double *param, size_t n);

double gaussian(double mean, double var);
double gaussian_pdf(double x, double mean, double var);
double gaussian_cdf(double x, double mean, double var);

double gamma(double shape, double rate);
double gamma_pdf(double x, double shape, double rate);
double gamma_cdf(double x, double shape, double rate);

double inverse_gamma(double shape, double scale);
double inverse_gamma_pdf(double x, double shape, double scale);
double inverse_gamma_cdf(double x, double shape, double scale);

// sample from an `n`-dimensional dirichlet in `dst`
void dirichlet(double *dst, double *param, size_t n);
double dirichlet_pdf(double *x, double *param, size_t n);