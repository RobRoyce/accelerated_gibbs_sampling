from collections import namedtuple
from scipy.stats import uniform, norm, gamma, invgamma, dirichlet
import numpy as np

"""Parameters of each generated continuous distr. pdf/cdf test.

Each pdf/cdf test consists of evaluating the C implementation of the
function of interest at `n` points in a given interval. 
 a: lower bound of the interval of test values
 b: upper bound of the interval of test values
 n: number of test values
 name: name of the test to be printed by the test program
 c_fn_name: name of the C function being tested
 args: additional arguments of the C function being tested
"""
TestParams = namedtuple('TestParams',
                        'a b n f name c_fn_name args'.split())

test_params_list = [
    TestParams(-1, 2, 100, uniform.pdf, 'uniform(0, 1) pdf',
        'uniform_pdf', (0, 1)),
    TestParams(-1, 2, 100, uniform.cdf, 'uniform(0, 1) cdf',
        'uniform_cdf', (0, 1)),

    TestParams(-10, 10, 1000, norm.pdf, 'gaussian(0, 1) pdf',
        'gaussian_pdf', (0, 1)),
    TestParams(-10, 20, 1000, lambda x: norm.pdf(x, 10, 2**.5),
        'gaussian(10, 2) pdf', 'gaussian_pdf', (10, 2)),
    TestParams(-10, 10, 1000, norm.cdf, 'gaussian(0, 1) cdf',
        'gaussian_cdf', (0, 1)),
    TestParams(-10, 20, 1000, lambda x: norm.cdf(x, 10, 2**.5),
        'gaussian(10, 2) cdf', 'gaussian_cdf', (10, 2)),

    TestParams(-1, 10, 1000, lambda x: gamma.pdf(x, 1),
        'gamma(1, 1) pdf', 'gamma_pdf', (1, 1)),
    TestParams(-1, 10, 1000, lambda x: gamma.pdf(x, 1.25, 0, 1),
        'gamma(1.25, 1) pdf', 'gamma_pdf', (1.25, 1)),
    TestParams(-1, 10, 1000, lambda x: gamma.pdf(x, 1.25, 0, 1/2),
        'gamma(1.25, 2) pdf', 'gamma_pdf', (1.25, 2)),
    TestParams(-1, 10, 1000, lambda x: gamma.cdf(x, 1),
        'gamma(1, 1) cdf', 'gamma_cdf', (1, 1)),
    TestParams(-1, 10, 1000, lambda x: gamma.cdf(x, 1.25, 0, 1),
        'gamma(1.25, 1) cdf', 'gamma_cdf', (1.25, 1)),
    TestParams(-1, 10, 1000, lambda x: gamma.cdf(x, 1.25, 0, 1/2),
        'gamma(1.25, 2) cdf', 'gamma_cdf', (1.25, 2)),

    TestParams(-1, 10, 1000, lambda x: invgamma.pdf(x, 1),
        'inverse_gamma(1, 1) pdf', 'inverse_gamma_pdf', (1, 1)),
    TestParams(-1, 10, 1000, lambda x: invgamma.pdf(x, 1.25, 0, 1),
        'inverse_gamma(1.25, 1) pdf', 'inverse_gamma_pdf', (1.25, 1)),
    TestParams(-1, 10, 1000, lambda x: invgamma.pdf(x, 1.25, 0, 2),
        'inverse_gamma(1.25, 2) pdf', 'inverse_gamma_pdf', (1.25, 2)),
    TestParams(-1, 10, 1000, lambda x: invgamma.cdf(x, 1),
        'inverse_gamma(1, 1) cdf', 'inverse_gamma_cdf', (1, 1)),
    TestParams(-1, 10, 1000, lambda x: invgamma.cdf(x, 1.25, 0, 1),
        'inverse_gamma(1.25, 1) cdf', 'inverse_gamma_cdf', (1.25, 1)),
    TestParams(-1, 10, 1000, lambda x: invgamma.cdf(x, 1.25, 0, 1/2),
        'inverse_gamma(1.25, 2) cdf', 'inverse_gamma_cdf', (1.25, 2)),
]

head = r"""#include <stdio.h>
#include <stdlib.h>

#include "../src/distrs.h"

#define EPS 1e-5

int test_fn(DTYPE *x, DTYPE *y, size_t n, DTYPE (*f)(DTYPE x), char *name)
{
    for(int i=0; i < n; i++) {
        if(abs(y[i] - f(x[i])) > EPS) {
            printf("%s failed: expected %f at %f, got %f\n",
                   name, y[i], x[i], f(x[i]));
            return 0;
        }
    }
    printf("%s as expected.\n", name);
    return 1;
}
"""

test_params_str = """
DTYPE x_{i}[] = {{{xs_str}}};
DTYPE y_{i}[] = {{{ys_str}}};
size_t n_{i} = {n};
char name_{i}[] = "{name}";
DTYPE f_{i}(DTYPE x) {{
    return {c_fn_name}(x, {args_str});
}}
"""

def make_param_strs(test_param):
    csv = lambda vals: ', '.join(map(str, vals))
    a, b, n, f, name, c_fn_name, args = test_param
    xs = a + (b - a) * np.random.rand(n)
    ys = f(xs)
    return dict(xs_str=csv(xs), ys_str=csv(ys), n=n, name=name,
                c_fn_name=c_fn_name, args_str=csv(args))

def make_main(n):
    call_str = 'test_fn(x_{0}, y_{0}, n_{0}, f_{0}, name_{0})'
    calls = (' &&\n' + ' '*13).join(call_str.format(i) for i in range(n))
    return 'int main()\n{{\n    return !({});\n}}'.format(calls)

if __name__ == '__main__':
    print(head)
    for i, test_params in enumerate(test_params_list):
        print(test_params_str.format(i=i, **make_param_strs(test_params)))
    print(make_main(len(test_params_list)))
