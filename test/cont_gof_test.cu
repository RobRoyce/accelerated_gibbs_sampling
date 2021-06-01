#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "../src/distrs.h"
#include "../src/utils.h"
#include <string.h>
#include <iostream>

/*
 * Critical value of the distribution of the test statistic in the Cramer-von
 * Mises test for large samples, with 99% significance.
 */
#define CRITICAL_VAL 2.3263

// sample size
#define N 100000

double sample[N];
int DEBUG = 0;

int double_comp(const void *a, const void *b) {
    return *((double *) a) - *((double *) b) < 0 ? -1 : 1;
}

int cramer_von_mises(double (*sampler)(), double (*cdf)(double), std::string name) {
    for (size_t i = 0; i < N; sample[i++] = sampler());
    qsort(sample, N, sizeof(double), double_comp);
    double statistic = 1.0 / 12 / N;
    for (size_t i = 1; i <= N; i++)
        statistic += square(cdf(sample[i - 1]) - (2.0 * i - 1) / 2 / N);
    int success = statistic < CRITICAL_VAL ? 1 : 0;

    if (DEBUG) {
        printf("%s %s Cramer-von Mises (statistic=%f).\n", name.c_str(),
               success ? "passed" : "failed", statistic);
    }

    return success;
}

double sampler_1() { return uniform(0, 1); }

double cdf_1(double x) { return uniform_cdf(x, 0, 1); }

double sampler_2() { return gaussian(0, 1); }

double cdf_2(double x) { return gaussian_cdf(x, 0, 1); }

double sampler_3() { return gaussian(2, 0.4); }

double cdf_3(double x) { return gaussian_cdf(x, 2, 0.4); }

double sampler_4() { return gamma(1, 1); }

double cdf_4(double x) { return gamma_cdf(x, 1, 1); }

double sampler_5() { return gamma(1.25, 2); }

double cdf_5(double x) { return gamma_cdf(x, 1.25, 2); }

double sampler_6() { return inverse_gamma(1, 1); }

double cdf_6(double x) { return inverse_gamma_cdf(x, 1, 1); }

double sampler_7() { return inverse_gamma(1.25, 2); }

double cdf_7(double x) { return inverse_gamma_cdf(x, 1.25, 2); }

int main(int argc, char **argv) {
    DEBUG = (argc > 1) && (strcmp(argv[1], "--debug") == 0) ? 1 : 0;

    srand(time(NULL));
    if (!(cramer_von_mises(sampler_1, cdf_1, "uniform(0, 1)") &&
          cramer_von_mises(sampler_2, cdf_2, "gaussian(0, 1)") &&
          cramer_von_mises(sampler_3, cdf_3, "gaussian(2, 0.4)") &&
          cramer_von_mises(sampler_4, cdf_4, "gamma(1, 1)") &&
          cramer_von_mises(sampler_5, cdf_5, "gamma(1.25, 2)") &&
          cramer_von_mises(sampler_6, cdf_6, "inverse_gamma(1, 1)") &&
          cramer_von_mises(sampler_7, cdf_7, "inverse_gamma(1.25, 2)"))) {
        printf("cont_gof_test: Success!\n");
        return 0;
    } else {
        printf("cont_gof_test: FAILED!\n");
    }
}
