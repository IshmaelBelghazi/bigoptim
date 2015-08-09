#include <stdio.h>
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "utils.h"
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"
#include "sag_step.h"

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
