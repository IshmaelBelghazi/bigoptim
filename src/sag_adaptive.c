#include <stdio.h>
#include <math.h>
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "utils.h"
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"
#include "sag_step.h"

// TODO(Ishmael): Consider using R math functions
const static int DEBUG = 0;
const static int sparse = 0;
const static double precision = 1.490116119384765625e-8;

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
SEXP C_sag_adaptive(SEXP w, SEXP Xt, SEXP y, SEXP lambda, SEXP Lmax,
                    SEXP Li, SEXP randVals, SEXP d, SEXP g, SEXP covered,
                    SEXP increasing) {
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



  /* Initializing dataset */
  Dataset train_set = { .Xt = REAL(Xt),
                        .y = REAL(y),
                        .randVals = REAL(randVals),
                        .nSamples = INTEGER(GET_DIM(Xt))[1],
                        .nVars = INTEGER(GET_DIM(Xt))[0],
                        .Lmax = REAL(Lmax),
                        .Li = REAL(Li),
                        .covered = INTEGER(covered),
                        .increasing = *INTEGER(increasing),
                        .sparse = sparse};
  
  /* Initialzing trainer */
  GlmTrainer trainer = {.lambda = *REAL(lambda),
                        .d = REAL(d),
                        .g = REAL(g),
                        .iter = 0,
                        .maxIter = INTEGER(GET_DIM(randVals))[0],
                        .precision = precision,
                        .step = _sag_adaptive_iteration};
  
   /* Initializing Model */
   GlmModel model = {.w = REAL(w), .loss = binomial_loss, .grad = binomial_grad};
  
  /*Error Checking*/
  if (train_set.nVars != INTEGER(GET_DIM(w))[0]) {
    error("w and Xt must have the same number of rows");
  }
  if (train_set.nSamples != INTEGER(GET_DIM(y))[0]) {
    error("number of columns of Xt must be the same as the number of rows in y");
  }
  if (train_set.nVars != INTEGER(GET_DIM(d))[0]) {
    error("w and d must have the same number of rows");
  }
  if (train_set.nSamples != INTEGER(GET_DIM(g))[0]) {
    error("w and g must have the same number of rows");
  }
  if (train_set.nSamples != INTEGER(GET_DIM(covered))[0]) {
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
  for(int i = 0; i < train_set.nSamples; i++) {
    if (train_set.covered[i] != 0) {
      nCovered++;
      Lmean += train_set.Li[i];
    }
  }
  
  if(nCovered > 0) {
    Lmean /= nCovered;
  }
  

  /* Do the O(n log n) initialization of the data structures
     will allow sampling in O(log(n)) time */
  int nextpow2 = pow(2, ceil(log2(train_set.nSamples)/log2(2)));
  int nLevels = 1 + (int)ceil(log2(train_set.nSamples));
  if (DEBUG) Rprintf("next power of 2 is: %d\n",nextpow2);
  if (DEBUG) Rprintf("nLevels = %d\n",nLevels);
  /* Counts number of descendents in tree */
  double * nDescendants = Calloc(nextpow2 * nLevels, double); 
  /* Counts number of descenents that are still uncovered */
  double * unCoveredMatrix = Calloc(nextpow2 * nLevels, double); 
  /* Sums Lipschitz constant of loss over descendants */
  double * LiMatrix = Calloc(nextpow2 * nLevels, double); 
  for(int i = 0; i < train_set.nSamples; i++) {
    nDescendants[i] = 1;
    if (train_set.covered[i]) {
        LiMatrix[i] = train_set.Li[i];
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

  /* Continuing dataset initialisation */
  train_set.Lmean = Lmean;
  train_set.nextpow2 = nextpow2;
  train_set.nDescendants = nDescendants;
  train_set.unCoveredMatrix = unCoveredMatrix;
  train_set.LiMatrix = LiMatrix;

  

  for(int k = 0; k < trainer.maxIter; k++) {
    // TODO(Ishmael): Add iteration for adaptive SAG
    
    trainer.step(&trainer, &model, &train_set);
  }

  if (sparse) {
    // TODO(Ishmael): SAG_LipschitzLS_logistic_BLAS.c line 315
  }

  /* Freeing allocated variables */
  Free(nDescendants);
  Free(unCoveredMatrix);
  Free(LiMatrix);

  /*=======\
  | Return |
  \=======*/
  
  /* Preparing return variables  */
  SEXP w_return = PROTECT(allocMatrix(REALSXP, train_set.nVars, 1)); nprot++;
  Memcpy(REAL(w_return), model.w, train_set.nVars);
  SEXP d_return = PROTECT(allocMatrix(REALSXP, train_set.nVars, 1)); nprot++;
  Memcpy(REAL(d_return), trainer.d, train_set.nVars);
  SEXP g_return = PROTECT(allocMatrix(REALSXP, train_set.nSamples, 1)); nprot++;
  Memcpy(REAL(g_return), trainer.g, train_set.nSamples);
  SEXP covered_return = PROTECT(allocMatrix(INTSXP, train_set.nSamples, 1)); nprot++;
  Memcpy(INTEGER(covered_return), train_set.covered, train_set.nSamples);

  /* Assigning variables to SEXP list */
  SEXP results = PROTECT(allocVector(VECSXP, 4)); nprot++;
  INC_APPLY(SEXP, SET_VECTOR_ELT, results, w_return, d_return, g_return, covered_return); // in utils.h
  /* Creating SEXP for list names */
  SEXP results_names = PROTECT(allocVector(STRSXP, 4)); nprot++;
  INC_APPLY_SUB(char *, SET_STRING_ELT, mkChar, results_names, "w", "d", "g", "covered");
  setAttrib(results, R_NamesSymbol, results_names);

  UNPROTECT(nprot);
  return results;
}
