#include "entrypoint-glm.h"
/*=========\
| BINOMIAL |
\=========*/

SEXP C_binomial_cost(SEXP Xt, SEXP y, SEXP w, SEXP lambda) {
  // Initializing protection counter
  int nprot = 0;
  int nSamples = INTEGER(GET_DIM(Xt))[1];
  int nVars = INTEGER(GET_DIM(Xt))[0];
  SEXP cost = PROTECT(allocVector(REALSXP, 1)); nprot++;

  *REAL(cost) = binomial_cost(REAL(Xt), REAL(y), REAL(w), *REAL(lambda),
                               nSamples, nVars);
  UNPROTECT(nprot);
  return cost;
}

SEXP C_binomial_cost_grad(SEXP Xt, SEXP y, SEXP w, SEXP lambda) {
  // Initializing Protection counter
  int nprot = 0;
  int nSamples = INTEGER(GET_DIM(Xt))[1];
  int nVars = INTEGER(GET_DIM(Xt))[0];
  /* Allocating grad SEXP */
  SEXP grad = PROTECT(allocMatrix(REALSXP, nVars, 1)); nprot++;
  memset(REAL(grad), 0.0, nVars * sizeof(double));
  binomial_cost_grad(REAL(Xt), REAL(y), REAL(w), *REAL(lambda),
                      nSamples, nVars,
                      REAL(grad));

  UNPROTECT(nprot);
  return grad;
}

/*=========\
| GAUSSIAN |
\=========*/

SEXP C_gaussian_cost(SEXP Xt, SEXP y, SEXP w, SEXP lambda) {
  // Initializing protection counter
  int nprot = 0;
  int nSamples = INTEGER(GET_DIM(Xt))[1];
  int nVars = INTEGER(GET_DIM(Xt))[0];
  SEXP cost = PROTECT(allocVector(REALSXP, 1)); nprot++;

  *REAL(cost) = gaussian_cost(REAL(Xt), REAL(y), REAL(w), *REAL(lambda),
                              nSamples, nVars);
  UNPROTECT(nprot);
  return cost;
}

SEXP C_gaussian_cost_grad(SEXP Xt, SEXP y, SEXP w, SEXP lambda) {
  // Initializing Protection counter
  int nprot = 0;
  int nSamples = INTEGER(GET_DIM(Xt))[1];
  int nVars = INTEGER(GET_DIM(Xt))[0];
  /* Allocating grad SEXP */
  SEXP grad = PROTECT(allocMatrix(REALSXP, nVars, 1)); nprot++;
  memset(REAL(grad), 0.0, nVars * sizeof(double));
  gaussian_cost_grad(REAL(Xt), REAL(y), REAL(w), *REAL(lambda),
                     nSamples, nVars,
                     REAL(grad));

  UNPROTECT(nprot);
  return grad;
}

/*============\
| EXPONENTIAL |
\============*/

SEXP C_exponential_cost(SEXP Xt, SEXP y, SEXP w, SEXP lambda) {
  // Initializing protection counter
  int nprot = 0;
  int nSamples = INTEGER(GET_DIM(Xt))[1];
  int nVars = INTEGER(GET_DIM(Xt))[0];
  SEXP cost = PROTECT(allocVector(REALSXP, 1)); nprot++;

  *REAL(cost) = exponential_cost(REAL(Xt), REAL(y), REAL(w), *REAL(lambda),
                              nSamples, nVars);
  UNPROTECT(nprot);
  return cost;
}

SEXP C_exponential_cost_grad(SEXP Xt, SEXP y, SEXP w, SEXP lambda) {
  // Initializing Protection counter
  int nprot = 0;
  int nSamples = INTEGER(GET_DIM(Xt))[1];
  int nVars = INTEGER(GET_DIM(Xt))[0];
  /* Allocating grad SEXP */
  SEXP grad = PROTECT(allocMatrix(REALSXP, nVars, 1)); nprot++;
  memset(REAL(grad), 0.0, nVars * sizeof(double));
  exponential_cost_grad(REAL(Xt), REAL(y), REAL(w), *REAL(lambda),
                     nSamples, nVars,
                     REAL(grad));

  UNPROTECT(nprot);
  return grad;
}

/*========\
| POISSON |
\========*/

SEXP C_poisson_cost(SEXP Xt, SEXP y, SEXP w, SEXP lambda) {
  // Initializing protection counter
  int nprot = 0;
  int nSamples = INTEGER(GET_DIM(Xt))[1];
  int nVars = INTEGER(GET_DIM(Xt))[0];
  SEXP cost = PROTECT(allocVector(REALSXP, 1)); nprot++;

  *REAL(cost) = poisson_cost(REAL(Xt), REAL(y), REAL(w), *REAL(lambda),
                              nSamples, nVars);
  UNPROTECT(nprot);
  return cost;
}

SEXP C_poisson_cost_grad(SEXP Xt, SEXP y, SEXP w, SEXP lambda) {
  // Initializing Protection counter
  int nprot = 0;
  int nSamples = INTEGER(GET_DIM(Xt))[1];
  int nVars = INTEGER(GET_DIM(Xt))[0];
  /* Allocating grad SEXP */
  SEXP grad = PROTECT(allocMatrix(REALSXP, nVars, 1)); nprot++;
  memset(REAL(grad), 0.0, nVars * sizeof(double));
  poisson_cost_grad(REAL(Xt), REAL(y), REAL(w), *REAL(lambda),
                    nSamples, nVars,
                    REAL(grad));

  UNPROTECT(nprot);
  return grad;
}
