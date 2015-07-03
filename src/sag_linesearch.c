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

const static int sparse = 0;
const static double precision = 1.490116119384765625e-8;
/**
 *     Logistic regression stochastic average gradient trainer
 *    
 *     @param w(p, 1) weights
 *     @param Xt(p, n) real feature matrix
 *     @param y(n, 1) {-1, 1} target matrix
 *     @param lambda scalar regularization parameters
 *     @param stepSizeType scalar constant step size
 *     @param iVals(max_iter, 1) sequence of examples to choose
 *     @param d(p, 1) initial approximation of average gradient
 *     @param g(n, 1) previous derivatives of loss
 *     @param covered(n, 1) whether the example has been visited
 *     @param stepSizeType scalar default is 1 to use 1/L, set to 2 to
 *     use 2/(L + n*myu)
 *     @return optimal weights (p, 1)
 */
SEXP C_sag_linesearch(SEXP w, SEXP Xt, SEXP y, SEXP lambda,
                      SEXP stepSize, SEXP iVals, SEXP d, SEXP g,
                      SEXP covered, SEXP stepSizeType) {
  // Initializing protection counter
  int nprot = 0;

  /*======\
  | Input |
  \======*/
  /* Initializing dataset */
  Dataset train_set = {.Xt = REAL(Xt),
                       .y = REAL(y),
                       .iVals = INTEGER(iVals),
                       .covered = INTEGER(covered),
                       .nCovered = 0,
                       .nSamples = INTEGER(GET_DIM(Xt))[1],
                       .nVars = INTEGER(GET_DIM(Xt))[0],
                       .sparse = sparse};

  /* Initializing trainer  */
  GlmTrainer trainer = {.lambda = *REAL(lambda),                        
                        .d = REAL(d),
                        .g = REAL(g),
                        .iter = 0,
                        .maxIter = INTEGER(GET_DIM(iVals))[0],
                        .stepSizeType = *INTEGER(stepSizeType),
                        .Li = REAL(stepSize),
                        .precision = precision,
                        .step = _sag_linesearch_iteration};
  
  /* Initializing Model */
  GlmModel model = {.w = REAL(w), .loss = logistic_loss, .grad = logistic_grad};

  
  // Mark deals with stepSizeType and xtx as optional arguments. This
  // makes sense in MATLAB. In R it is simpler to pass the default
  // argument in R when using .Call rather than use .Extern
  /* double * xtx = Calloc(train_set.nSamples, double); */

  /*===============\
  | Error Checking |
  \===============*/
  if ( train_set.nVars != INTEGER(GET_DIM(w))[0]) {
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
  // TODO(Ishmael): SAGlineSearch_logistic_BLAS.c line 72
  /* if (sparse && alpha * lambda == 1) { // BUG(Ishmael): BUG is mark's */
  /*       			       // code alpha is not declared yet. */
  /*   error("Sorry, I don't like it when Xt is sparse and alpha*lambda=1\n"); */
  /* } */
  /* Allocate Memory Needed for Lazy Update */
  if (sparse) {
    // TODO(Ishmael): SAGlineSearch_logistic_BLAS.c line 82
  }

  
  for(int i = 0; i < train_set.nSamples; i++) {
    if (train_set.covered[i]!=0) train_set.nCovered++;
  }
  
  /* for (int i = 0; i < train_set.nSamples; i++) { */
  /*   if (sparse) { */
  /*     // TODO(Ishmael): SAGlineSearch_logistic_BLAS.c line 103 */
  /*   } else { */
  /*     xtx[i] = F77_CALL(ddot)(&train_set.nVars, &Xt[i * train_set.nVars], &one, &Xt[i * train_set.nVars], &one); */
  /*   } */
  /* } */

  /*============================\
  | Stochastic Average Gradient |
  \============================*/
  for (trainer.iter = 0; trainer.iter  < trainer.maxIter; trainer.iter++) {
    trainer.step(&trainer, &model, &train_set);
  }

  /* Freeing Allocated variables */
  /* Free(xtx); */

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

