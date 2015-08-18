#include <stdio.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "Matrix.h"
#include "cholmod.h"
#include "sag_constant_marks.h"

#define R_TRACE( x, ... ) Rprintf(" TRACE @ %s:%d \t" x "\n", __FILE__, __LINE__, ##__VA_ARGS__)
const static int DEBUG = 0;

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
SEXP C_sag_constant_mark(SEXP w_s, SEXP Xt_s, SEXP y_s, SEXP lambda_s,
                         SEXP stepSize_s, SEXP iVals_s, SEXP d_s, SEXP g_s,
                         SEXP covered_s) {
  Rprintf("In C file\n");
  // Initializing protection counter
  int nprot = 0;
  /* Variables  */
  int k, nSamples, maxIter, sparse = 1, *iVals, *covered, *lastVisited;
  long i, j;
  int nVars,
      one = 1; // With long clang complains about pointer type incompability

  int *jc, *ir;

  double *w, *Xt, *y, lambda, alpha, innerProd, sig,
      c = 1, *g, *d, nCovered = 0, *cumSum, scaling;

  /*======\
  | Input |
  \======*/
  Rprintf("Accessing cholmod struct\n");
  /* Accessing dgCMatrix object*/
  CHM_SP cXt = AS_CHM_SP(Xt_s);
  jc = cXt->p;
  ir = cXt->i;
  // Xt = REAL(Xt_s);
  Xt = cXt->x;

  w = REAL(w_s);
  y = REAL(y_s);
  lambda = *REAL(lambda_s);
  alpha = *REAL(stepSize_s);
  iVals = INTEGER(iVals_s);
  if (DEBUG)
    Rprintf("iVals[0]: %d\n", iVals[0]);
  d = REAL(d_s);
  g = REAL(g_s);
  covered = INTEGER(covered_s);
  if (DEBUG)
    Rprintf("covered[0]: %d\n", covered[0]);
  /* Compute sizes */
  nSamples = cXt->ncol; // INTEGER(GET_DIM(Xt_s))[1];
  nVars = cXt->nrow;    // INTEGER(GET_DIM(Xt_s))[0];
  maxIter = INTEGER(GET_DIM(iVals_s))[0];
  if (DEBUG)
    Rprintf("nSamples: %d\n", nSamples);
  if (DEBUG)
    Rprintf("nVars: %d\n", nVars);
  if (DEBUG)
    Rprintf("maxIter: %d\n", maxIter);

  /*===============\
  | Error Checking |
  \===============*/
  if (nVars != INTEGER(GET_DIM(w_s))[0]) {
    error("w and Xt must have the same number of rows");
  }
  if (nSamples != INTEGER(GET_DIM(y_s))[0]) {
    error(
        "number of columns of Xt must be the same as the number of rows in y");
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
    error("Sorry, I don't like it when Xt is sparse and alpha*lambda=1\n");
  }

  /*==============================\
  | Stochastic Average Gradient   |
  \==============================*/
  /* Allocate Memory Needed for lazy update */
  if (sparse) {
    // TODO(Ishmael): If (sparse) line 72 in SAG_logistic_BLAS
    lastVisited = Calloc(nVars, int);
    cumSum = Calloc(maxIter, double);
  }
  for (int i = 0; i < nSamples; i++) {
    if (covered[i] != 0)
      nCovered++;
  }

  for (k = 0; k < maxIter; k++) {
    /* Select next training example */
    i = iVals[k] - 1;
    /* Compute current values of needed parameters */
    if (sparse && k > 0) {
      for (j = jc[i]; j < jc[i + 1]; j++) {
        if (lastVisited[ir[j]] == 0) {
          w[ir[j]] -= d[ir[j]] * cumSum[k - 1];
        } else {
          w[ir[j]] -=
              d[ir[j]] * (cumSum[k - 1] - cumSum[lastVisited[ir[j]] - 1]);
        }
        lastVisited[ir[j]] = k;
      }
    }

    /* Compute derivative of loss */
    if (sparse) {
      innerProd = 0;
      for (j = jc[i]; j < jc[i + 1]; j++)
        innerProd += w[ir[j]] * Xt[j];
      innerProd *= c;
    } else {
      innerProd = F77_CALL(ddot)(&nVars, w, &one, &Xt[nVars * i], &one);
    }

    sig = -y[i] / (1 + exp(y[i] * innerProd));

    /* Update direction */
    if (sparse) {
      for (j = jc[i]; j < jc[i + 1]; j++)
        d[ir[j]] += Xt[j] * (sig - g[i]);
    } else {
      scaling = sig - g[i];
      F77_CALL(daxpy)(&nVars, &scaling, &Xt[i * nVars], &one, d, &one);
    }

    /* Store derivative of loss */
    g[i] = sig;
    /* Update the number of examples that we have seen */
    if (covered[i] == 0) {
      covered[i] = 1;
      nCovered++;
    }

    /* Update parameters */
    if (sparse) {
      c *= 1 - alpha * lambda;
      if (k == 0)
        cumSum[0] = alpha / (c * nCovered);
      else
        cumSum[k] = cumSum[k - 1] + alpha / (c * nCovered);
    } else {
      scaling = 1 - alpha * lambda;
      F77_CALL(dscal)(&nVars, &scaling, w, &one);
      scaling = -alpha / nCovered;
      F77_CALL(daxpy)(&nVars, &scaling, d, &one, w, &one);
    }
  }
  if (sparse) {
    for (j = 0; j < nVars; j++) {
      if (lastVisited[j] == 0) {
        w[j] -= d[j] * cumSum[maxIter - 1];
      } else {
        w[j] -= d[j] * (cumSum[maxIter - 1] - cumSum[lastVisited[j] - 1]);
      }
    }
    scaling = c;
    F77_CALL(dscal)(&nVars, &scaling, w, &one);
    Free(lastVisited);
    Free(cumSum);
  }

  /*=======\
  | Return |
  \=======*/
  /* Preparing return variables  */
  SEXP w_return = PROTECT(allocMatrix(REALSXP, nVars, 1));
  nprot++;
  Memcpy(REAL(w_return), w, nVars);
  SEXP d_return = PROTECT(allocMatrix(REALSXP, nVars, 1));
  nprot++;
  Memcpy(REAL(d_return), d, nVars);
  SEXP g_return = PROTECT(allocMatrix(REALSXP, nSamples, 1));
  nprot++;
  Memcpy(REAL(g_return), g, nSamples);
  SEXP covered_return = PROTECT(allocMatrix(INTSXP, nSamples, 1));
  nprot++;
  Memcpy(INTEGER(covered_return), covered, nSamples);

  /* Assigning variables to list */
  SEXP results = PROTECT(allocVector(VECSXP, 4));
  nprot++;
  SET_VECTOR_ELT(results, 0, w_return);
  SET_VECTOR_ELT(results, 1, d_return);
  SET_VECTOR_ELT(results, 2, g_return);
  SET_VECTOR_ELT(results, 3, covered_return);
  /* Setting list names */
  SEXP results_names = PROTECT(allocVector(STRSXP, 4));
  nprot++;
  const char *names[4] = {"w", "d", "g", "covered"};
  for (int i = 0; i < 4; i++) {
    SET_STRING_ELT(results_names, i, mkChar(names[i]));
  }
  setAttrib(results, R_NamesSymbol, results_names);

  // SEXP results = PROTECT(allocVector(VECSXP, 3)); nprot++;
  UNPROTECT(nprot);
  return results;
}
