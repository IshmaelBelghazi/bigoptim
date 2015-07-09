#include <stdio.h>
#include <math.h>
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "glm_models.h"
#include "utils.h"

// TODO(Ishmael): Consider using R math functions
const static int DEBUG = 0;
const static int one = 1;
const static int sparse = 0;
/**
 *   Stochastic Average Gradient Descent with line-search and adaptive
 *   lipschitz sampling
 *   
 *   @param w_s (p, 1) real weights
 *   @param Xt_s (p, n) real features Matrix
 *   @param y_s (m, 1) {-1, 1} targets Matrix
 *   @param lambda_s scalar regularization parameter
 *   @param Lmax_s scalar initial approximation of global Lipschitz constants
 *   @param Li_s (n, 1) initial approximation of inidividual lipschitz constants
 *   @param randVals (maxiter, 2) - sequence of random values for the
 *   algorithm to use
 *   @param d_s (p, 1) initial approximation of average gradient
 *   @param g_s (n, 1) previousd derivatives of loss
 *  
 *   @param covered_s  d(p,1) initial approximation of average gradient (should be sum of previous gradients)
 *   @param increasing_s  scalar default is 1 to allow the Lipscthiz constants to increase, set to 0 to only allow them to decrease
 *     
 *   @return optimal weights (p, 1)
 */
SEXP C_sag_adaptive(SEXP w_s, SEXP Xt_s, SEXP y_s, SEXP lambda_s, SEXP Lmax_s,
                    SEXP Li_s, SEXP randVals_s, SEXP d_s, SEXP g_s, SEXP covered_s,
                    SEXP increasing_s) {
  // initializing protection counter
  int nprot = 0;
  /* Variables */
  // TODO(Ishmael): This is messy. Clean it. 
  // int temp;
  // int  * lastVisited;
  // int i, j;
  // size_t * jc,* ir;
  
  // double c=1;
  // double * cumSum;
  
  /*======\
  | Input |
  \======*/

  Dataset train_set = { .Xt = REAL(Xt_s),}
  
  double * w = REAL(w_s);
  double * Xt = REAL(Xt_s);
  double * y = REAL(y_s);
  double lambda = *REAL(lambda_s);
  double * Lmax = REAL(Lmax_s);
  double * Li = REAL(Li_s);
  double * randVals = REAL(randVals_s);
  double * d = REAL(d_s);
  double * g = REAL(g_s);
  int * covered = INTEGER(covered_s);
  int increasing = *INTEGER(increasing_s);

  /* Compute Sizes */
  int nSamples = INTEGER(GET_DIM(Xt_s))[1];
  int nVars = INTEGER(GET_DIM(Xt_s))[0];
  int maxIter = INTEGER(GET_DIM(randVals_s))[0];
  if (DEBUG) Rprintf("nSamples: %d\n", nSamples);
  if (DEBUG) Rprintf("nVars: %d\n", nVars);
  if (DEBUG) Rprintf("maxIter: %d\n", maxIter);

  double precision = 1.490116119384765625e-8;
  double * xtx = Calloc(nSamples, double);


  /* Error Checking */
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
  // TODO(Ishmael): SAG_LipschitzLS_logistic_BLAS line 78
  /* if (sparse && alpha * lambda == 1) { */
  /*   error("Sorry, I don't like it when Xt is sparse and alpha*lambda=1\n"); */
  /* } */
  /*============================\
  | Stochastic Average Gradient |
  \============================*/
  /* Allocate memory needed for lazy updates*/
  if (sparse) {
    // TODO(Ishmael): SAG_LipschitzLS_logistic_BLAS line 89
  }
   
  /* Compute mean of covered variables */
  double Lmean = 0;
  double nCovered = 0;
  for(int i = 0; i < nSamples; i++) {
    if (covered[i] != 0) {
      nCovered++;
      Lmean += Li[i];
    }
  }
  
  if(nCovered > 0) {
    Lmean /= nCovered;
  }
  
  for (int i = 0; i < nSamples; i++) {
    if (sparse) {
      // TODO(Ishmael): SAGlineSearch_logistic_BLAS.c line 103
    } else {
      // TODO(Ishmael): use a higher level BLAS OPERATION
      xtx[i] = F77_CALL(ddot)(&nVars, &Xt[i * nVars], &one, &Xt[i * nVars], &one);
    }
  }
  /* Do the O(n log n) initialization of the data structures
     will allow sampling in O(log(n)) time */
  int nextpow2 = pow(2, ceil(log2(nSamples)/log2(2)));
  int nLevels = 1 + (int)ceil(log2(nSamples));
  if (DEBUG) Rprintf("next power of 2 is: %d\n",nextpow2);
  if (DEBUG) Rprintf("nLevels = %d\n",nLevels);
  /* Counts number of descendents in tree */
  double * nDescendants = Calloc(nextpow2 * nLevels, double); 
  /* Counts number of descenents that are still uncovered */
  double * unCoveredMatrix = Calloc(nextpow2 * nLevels, double); 
  /* Sums Lipschitz constant of loss over descendants */
  double * LiMatrix = Calloc(nextpow2 * nLevels, double); 
  for(int i = 0; i < nSamples; i++) {
    nDescendants[i] = 1;
    if (covered[i]) {
        LiMatrix[i] = Li[i];
    } else {
      unCoveredMatrix[i] = 1;
    }
    
  }
  int levelMax = nextpow2;
  for (int level = 1; level < nLevels; level++) {
    levelMax = levelMax/2;
    for(int i = 0; i < levelMax; i++) {
      nDescendants[i + nextpow2 * level] = nDescendants[ 2 * i + nextpow2 * (level - 1)] +
                                           nDescendants[ 2 * i + 1 + nextpow2 * (level - 1)];
      LiMatrix[i + nextpow2 * level] = LiMatrix[2 * i + nextpow2 * (level - 1)] +
                                       LiMatrix[ 2 * i + 1 + nextpow2 * (level - 1)];
      unCoveredMatrix[i + nextpow2 * level] = unCoveredMatrix[2 * i + nextpow2 * (level - 1)] +
                                              unCoveredMatrix[2 * i + 1 + nextpow2 * (level - 1)];
    }
  }

  for(int k = 0; k < maxIter; k++) {
    // TODO(Ishmael): Add iteration for adaptive SAG
  }

  if (sparse) {
    // TODO(Ishmael): SAG_LipschitzLS_logistic_BLAS.c line 315
  }

  /* Freeing allocated variables */
  Free(xtx);
  Free(nDescendants);
  Free(unCoveredMatrix);
  Free(LiMatrix);

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
  // SEXP results = PROTECT(allocVector(VECSXP, 3)); nprot++;
  UNPROTECT(nprot);

  return results;
}
