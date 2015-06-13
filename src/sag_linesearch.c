#include <stdio.h>
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>

const int DEBUG = 0;

/**
 *     Logistic regression stochastic average gradient trainer
 *    
 *     @param w_s(p, 1) weights
 *     @param Xt_s(p, n) real fature matrix
 *     @param y_s(n, 1) {-1, 1} target matrix
 *     @param lambda_s scalar regularization parameters
 *     @param stepSizeType_s scalar constant step size
 *     @param iVals_s(max_iter, 1) sequence of examples to choose
 *     @param d_s(p, 1) initial approximation of average gradient
 *     @param g_s(n, 1) previous derivatives of loss
 *     @param covered_s(n, 1) whether the example has been visited
 *     @param stepSizeType_s scalar default is 1 to use 1/L, set to 2 to
 *     use 2/(L + n*myu)
 *     @param xtx_s squared norm of features   
 *     @return optimal weights (p, 1)
 */
SEXP SAG_logistic(SEXP w_s, SEXP Xt_s, SEXP y_s, SEXP lambda_s,
                  SEXP stepSize_s, SEXP iVals_s, SEXP d_s, SEXP g_s,
                  SEXP covered_s, SEXP stepSizeType_s, SEXP xtx_s) {
  // Initializing protection counter
  int nprot = 0;
  /* Variables  */
  int k, nSamples, maxIter, sparse = 0, * iVals, * covered, * lastVisited;
  int temp, stepSizeType;
  
  long i, j;
  int nVars, one = 1; // With long clang complains about pointer type incompability

  size_t * jc, * ir;

  double * w, * Xt, * y, lambda, * Li, alpha, innerProd, sig,
    c=1, *g, *d, nCovered=0, * cumsum, fi, fi_new, gg, precision, scaling, wtx, * xtx;

  /*======\
  | Input |
  \======*/

  w = REAL(w_s);
  Xt = REAL(Xt_s);
  y = REAL(y_s);
  lambda = *REAL(lambda_s);
  Li = *REAL(stepSize_s);
  iVals = INTEGER(iVals_s);
  if (DEBUG) Rprintf("iVals[0]: %d\n", iVals[0]);
  d = REAL(d_s);
  g = REAL(g_s);
  covered = INTEGER(covered_s);
  if (DEBUG) Rprintf("covered[0]: %d\n", covered[0]);
  // Mark deal with stepSizeType and xtx as optional arguments. This
  // makes sense in MATLAB. In R it is simpler to pass the default
  // argument in R when using .Call rather than use .Extern
  stepSizeType = * INTEGER(stepSizeType_s);
  // TODO(Ishmael): Consider where to handle xtx
  /* Compute sizes */
  nSamples = INTEGER(GET_DIM(Xt_s))[1];
  nVars = INTEGER(GET_DIM(Xt_s))[0];
  maxIter = INTEGER(GET_DIM(iVals_s))[0];
  precision = 1.490116119384765625e-8;

  /*===============\
  | Error Checking |
  \===============*/
  if (nVars != INTEGER(GET_DIM(w_s))[0]) {
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
  // TODO(Ishmael): SAGlineSearch_logistic_BLAS.c line 72
  if (sparse && alpha * lambda == 1) { // BUG(Ishmael): BUG is mark's
				       // code alpha is not declared yet.
    error("Sorry, I don't like it when Xt is sparse and alpha*lambda=1\n");
  }
  /* Allocate Memory Needed for Lazy Update */
  if (sparse) {
    // TODO(Ishmael): SAGlineSearch_logistic_BLAS.c line 82
  }
  
  if (DEBUG) Rprintf("nSamples: %d\n", nSamples);
  if (DEBUG) Rprintf("nVars: %d\n", nVars);
  if (DEBUG) Rprintf("maxIter: %d\n", maxIter);
  // FIXME(Ishmael): Use size_t instead of int for indexing
  for(int i = 0; i < nSamples; i++) {
    if (covered[i]!=0) nCovered++;
  }
  for (int i = 0; i < nSamples; i++) {
    if (sparse) {
      // TODO(Ishmael): SAGlineSearch_logistic_BLAS.c line 103
    } else {
      xtx[i] = F77_CALL(ddot)(&nVars, &Xt[i * nVars], &one, &Xt[i * nVars], &one);
    }
  }

  /*============================\
  | Stochastic Average Gradient |
  \============================*/
  for (int k = 0; k < maxIter; k++) {
    /* Select next training example */
    i = iVals[k] - 1;
    if (sparse && k > 0) {
      //TODO(Ishmael): SAGlineSearch_logistic_BLAS.c line 132
    }
       
  }



  
  
  return NULL;
}
