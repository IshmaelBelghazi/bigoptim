#include <stdio.h>
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "glm_models.h"

/* Constant */
const static int DEBUG = 0;
const static int one = 1;
const static int sparse = 0;

/* Prototypes */
static inline void _sag_constant_iteration(GlmModel * model, double * w,
                                           double * Xt, double * y,
                                           double lambda, double alpha,
                                           int * iVals, double * d, double * g,
                                           int * covered, double * nCovered,
                                           int nSamples, int nVars,
                                           int maxIter, int k);
/*============\
| entry-point |
\============*/

/**
 * Logistic regression stochastic average gradient trainer
 *
 * @param w_s(p, 1) weights
 * @param Xt_s(p, n) real fature matrix
 * @param y_s(n, 1) {-1, 1} target matrix
 * @param lambda_s scalar regularization parameter
 * @param stepSize_s scalar constant step size
 * @param iVals_s(max_iter, 1) sequence of examples to choose
 * @param d_s(p, 1) initial approximation of average gradient
 * @param g_s(n, 1) previous derivatives of loss
 * @param covered_s(n, 1) whether the example has been visited
 * @return optimal weights (p, 1)
 */
SEXP C_sag_constant(SEXP w_s, SEXP Xt_s, SEXP y_s, SEXP lambda_s,
                    SEXP stepSize_s, SEXP iVals_s, SEXP d_s, SEXP g_s,
                    SEXP covered_s) {
  // Initializing garbage collection protection counter
  int nprot = 0;
  /* Variables  */
  // int * lastVisited;
  // int * jc, * ir;
  // double c=1, * cumsum;

  /*======\
  | Input |
  \======*/
  double * w = REAL(w_s);
  double * Xt = REAL(Xt_s);
  double * y = REAL(y_s);
  double lambda = *REAL(lambda_s);
  double alpha = *REAL(stepSize_s);
  int * iVals = INTEGER(iVals_s);
  if (DEBUG) Rprintf("iVals[0]: %d\n", iVals[0]);
  double * d = REAL(d_s);
  double * g = REAL(g_s);
  int * covered = INTEGER(covered_s);
  if (DEBUG) Rprintf("covered[0]: %d\n", covered[0]);
  /* Compute sizes */
  int nSamples = INTEGER(GET_DIM(Xt_s))[1];
  int nVars = INTEGER(GET_DIM(Xt_s))[0];
  int maxIter = INTEGER(GET_DIM(iVals_s))[0];
  if (DEBUG) Rprintf("nSamples: %d\n", nSamples);
  if (DEBUG) Rprintf("nVars: %d\n", nVars);
  if (DEBUG) Rprintf("maxIter: %d\n", maxIter);
  
  /*======\
  | Model |
  \======*/
  // TODO(Ishmael): Model Dispatch should go here
  GlmModel model = {.loss=logistic_loss, .grad=logistic_grad};
  
  /*===============\
  | Error Checking |
  \===============*/
  if (nVars != (int)INTEGER(GET_DIM(w_s))[0]) {
    error("w and Xt must have the same number of rows");
  }
  if (nSamples != INTEGER(GET_DIM(y_s))[0]) {
    error("number of columns of Xt must be the same as the number of rows in y");
  }
  if (nVars != INTEGER(GET_DIM(d_s))[0]) {
    error("w and d must have the same number of rows");
  }
  if (nSamples != INTEGER(GET_DIM(g_s))[0]) {
    error("w and g must have the same number of rows");
  }
  if (nSamples != INTEGER(GET_DIM(covered_s))[0]) {
    error("covered and y must hvae the same number of rows");
  }
  // TODO(Ishmael): SAG_logistic_BLAS line 62
  if (sparse && alpha * lambda == 1) {
    error("sorry, I don't like it when Xt is sparse and alpha*lambda=1\n");
  }
  
  /*==============================\
  | Stochastic Average Gradient   |
  \==============================*/
  /* Allocate Memory Needed for lazy update */
  if (sparse) {
  // TODO(Ishmael): If (sparse) line 72 in SAG_logistic_BLAS
  }
  double nCovered = 0;
  for (int i = 0; i < nSamples; i++) {
    if (covered[i] != 0) nCovered++;
  }
  
  for (int k = 0; k < maxIter; k++) {
    // Runing Iteration
    _sag_constant_iteration(&model, w, Xt, y, lambda, alpha,
                            iVals, d, g, covered, &nCovered,
                            nSamples, nVars, maxIter, k);
    
  }
  if (sparse) {
    // TODO(Ishmael): Line 153 in SAG_logistic_BLAS
  }

  /*=======\
  | Return |
  \=======*/
  /* Preparing return variables  */
  SEXP w_return = PROTECT(allocMatrix(REALSXP, nVars, 1)); nprot++;
  Memcpy(REAL(w_return), w, nVars);
  SEXP d_return = PROTECT(allocMatrix(REALSXP, nVars, 1)); nprot++;
  Memcpy(REAL(d_return), d, nVars);
  SEXP g_return = PROTECT(allocMatrix(REALSXP, nSamples, 1)); nprot++;
  Memcpy(REAL(g_return), g, nSamples);
  SEXP covered_return = PROTECT(allocMatrix(INTSXP, nSamples, 1)); nprot++;
  Memcpy(INTEGER(covered_return), covered, nSamples);

  /* Assigning variables to list */
  SEXP results = PROTECT(allocVector(VECSXP, 4)); nprot++;
  SET_VECTOR_ELT(results, 0, w_return);
  SET_VECTOR_ELT(results, 1, d_return);
  SET_VECTOR_ELT(results, 2, g_return);
  SET_VECTOR_ELT(results, 3, covered_return);
  /* Setting list names */
  SEXP results_names = PROTECT(allocVector(STRSXP, 4)); nprot++;
  const char * names[4] = {"w", "d", "g", "covered"};
  for (int i = 0; i < 4; i++) {
  SET_STRING_ELT(results_names, i, mkChar(names[i]));
  }
  setAttrib(results, R_NamesSymbol, results_names);

  UNPROTECT(nprot);
  return results;
}


static inline void _sag_constant_iteration(GlmModel * model, double * w, double * Xt,
                                           double * y, double lambda, double alpha,
                                           int * iVals, double * d, double * g,
                                           int * covered, double * nCovered,
                                           int nSamples, int nVars, int maxIter, int k) {
  // TODO(Ishmael): Rename k
  
  /* Select next training example */
  int i = iVals[k] - 1;
  /* Compute current values of needed parameters */
  if (sparse && k > 0) {
    //TODO(Ishmael): Line 91 in SAG_logistic_BLAS
  }
    
  /* Compute derivative of loss */
  double innerProd = 0;
  if (sparse) {
    //TODO(Ishmael): Line 104 in SAG_LOGISTIC_BLAS
  } else {
    innerProd = F77_CALL(ddot)(&nVars, w, &one, &Xt[nVars*i], &one);
  }

  double sig = model->grad(y[i], innerProd);

  /* Update direction */
  double scaling = 0;
  if (sparse) {
    // TODO(Ishmael): Line 117 in SAG_logistic_BLAS
  } else {
    scaling = sig - g[i];
    F77_CALL(daxpy)(&nVars, &scaling, &Xt[i * nVars], &one, d, &one);
  }

  /* Store derivative of loss */
  g[i] = sig;
  /* Update the number of examples that we have seen */
  if (covered[i] == 0) {
    covered[i] = 1;
    (*nCovered)++;
  }

  /* Update parameters */
  if (sparse) {
    // TODO(Ishmael): Line 135 in SAG_logistic_BLAS
  } else {
    scaling = 1 - alpha * lambda;
    F77_CALL(dscal)(&nVars, &scaling, w, &one);
    scaling = -alpha/(*nCovered);
    F77_CALL(daxpy)(&nVars, &scaling, d, &one, w, &one);
  }
}
