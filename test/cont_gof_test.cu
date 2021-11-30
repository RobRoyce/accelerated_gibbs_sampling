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

DTYPE sample[N];

int DTYPE_comp(const void *a, const void *b) {
    return *((DTYPE *) a) - *((DTYPE *) b) < 0 ? -1 : 1;
}

int cramer_von_mises(DTYPE (*sampler)(), DTYPE (*cdf)(DTYPE), std::string name) {
    for (size_t i = 0; i < N; sample[i++] = sampler());
    qsort(sample, N, sizeof(DTYPE), DTYPE_comp);
    DTYPE statistic = 1.0 / 12 / N;
    for (size_t i = 1; i <= N; i++)
        statistic += square(cdf(sample[i - 1]) - (2.0 * i - 1) / 2 / N);
    int success = statistic < CRITICAL_VAL ? 1 : 0;
    printf("%s %s Cramer-von Mises (statistic=%f).\n", name.c_str(),
           success ? "passed" : "failed", statistic);

    return success;
}

DTYPE sampler_1() { return uniform(0, 1); }
DTYPE cdf_1(DTYPE x) { return uniform_cdf(x, 0, 1); }

DTYPE sampler_2() { return gaussian(0, 1); }
DTYPE cdf_2(DTYPE x) { return gaussian_cdf(x, 0, 1); }

DTYPE sampler_3() { return gaussian(2, 0.4); }
DTYPE cdf_3(DTYPE x) { return gaussian_cdf(x, 2, 0.4); }

DTYPE sampler_4() { return gamma(1, 1); }
DTYPE cdf_4(DTYPE x) { return gamma_cdf(x, 1, 1); }

DTYPE sampler_5() { return gamma(1.25, 2); }
DTYPE cdf_5(DTYPE x) { return gamma_cdf(x, 1.25, 2); }

DTYPE sampler_6() { return inverse_gamma(1, 1); }
DTYPE cdf_6(DTYPE x) { return inverse_gamma_cdf(x, 1, 1); }

DTYPE sampler_7() { return inverse_gamma(1.25, 2); }
DTYPE cdf_7(DTYPE x) { return inverse_gamma_cdf(x, 1.25, 2); }

int main(int argc, char **argv) {

    srand(time(NULL));
    if (!(cramer_von_mises(sampler_1, cdf_1, "uniform(0, 1)") &&
          cramer_von_mises(sampler_2, cdf_2, "gaussian(0, 1)") &&
          cramer_von_mises(sampler_3, cdf_3, "gaussian(2, 0.4)") &&
          cramer_von_mises(sampler_4, cdf_4, "gamma(1, 1)") &&
          cramer_von_mises(sampler_5, cdf_5, "gamma(1.25, 2)") &&
          cramer_von_mises(sampler_6, cdf_6, "inverse_gamma(1, 1)") &&
          cramer_von_mises(sampler_7, cdf_7, "inverse_gamma(1.25, 2)"))) {
        printf("cont_gof_test ----------------------------------- FAILED!\n");
    } else {
        printf("cont_gof_test ----------------------------------- SUCCESS!\n");
    }
    return 0;
}
