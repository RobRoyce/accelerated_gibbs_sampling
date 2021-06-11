#include "gmm.h"


void allocDirichletParam(DTYPE **dirichletParam, DTYPE param, size_t k) {
    gpuErrchk(cudaMallocManaged(dirichletParam, k * sizeof(DTYPE)));
    for (int i = 0; i < k; i++)
        (*dirichletParam)[i] = param;
}

void freeDirichletParam(DTYPE *dirichletParam) {
    cudaFree(dirichletParam);
}

void randInitGmmParams(struct GMMParams *params, size_t n, size_t k, struct GMMPrior prior) {
    DTYPE *dirichletParam = nullptr;

    allocDirichletParam(&dirichletParam, prior.dirichletPrior, k);
    dirichlet(params->weights, dirichletParam, k);

    for (int j = 0; j < k; j++) {
        params->means[j] = gaussian(prior.meansMeanPrior, prior.meansVarPrior);
        params->vars[j] = inverse_gamma(prior.varsShapePrior, prior.varsScalePrior);
    }

    for (int i = 0; i < n; i++) {
        params->zs[i] = categorical(params->weights, k);
    }

    freeDirichletParam(dirichletParam);
}
