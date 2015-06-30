#include <stdio.h>
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"
#include "sag_step.h"

/* Constant */
const static int sparse = 0;


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
  /*======\
  | Input |
  \======*/
  
  /* Initializing dataset */
  Dataset train_set = {.Xt = REAL(Xt_s),
                       .y = REAL(y_s),
                       .iVals = INTEGER(iVals_s),
                       .covered = INTEGER(covered_s),
                       .nCovered = 0,
                       .nSamples = INTEGER(GET_DIM(Xt_s))[1],
                       .nVars = INTEGER(GET_DIM(Xt_s))[0],
                       .sparse = sparse};

  /* Initializing Trainer */
  GlmTrainer trainer = {.lambda = *REAL(lambda_s),
                         .alpha = *REAL(stepSize_s),
                         .d = REAL(d_s),
                         .g = REAL(g_s),
                         .iter = 0,
                         .maxIter = INTEGER(GET_DIM(iVals_s))[0],
                         .step = _sag_constant_iteration};
  /* Initializing Model */
  // TODO(Ishmael): Model Dispatch should go here
  GlmModel model = {.w = REAL(w_s), .loss=logistic_loss, .grad=logistic_grad};

  /*===============\
  | Error Checking |
  \===============*/
  if (train_set.nVars != INTEGER(GET_DIM(w_s))[0]) {
    error("w and Xt must have the same number of rows");
  }
  if (train_set.nSamples != INTEGER(GET_DIM(y_s))[0]) {
    error("number of columns of Xt must be the same as the number of rows in y");
  }
  if (train_set.nVars != INTEGER(GET_DIM(d_s))[0]) {
    error("w and d must have the same number of rows");
  }
  if (train_set.nSamples != INTEGER(GET_DIM(g_s))[0]) {
    error("w and g must have the same number of rows");
  }
  if (train_set.nSamples != INTEGER(GET_DIM(covered_s))[0]) {
    error("covered and y must have the same number of rows");
  }
  // TODO(Ishmael): SAG_logistic_BLAS line 62
  if (train_set.sparse && trainer.alpha * trainer.lambda == 1) {
    error("sorry, I don't like it when Xt is sparse and alpha*lambda=1\n");
  }
  
  /*==============================\
  | Stochastic Average Gradient   |
  \==============================*/
  /* Allocate Memory Needed for lazy update */
  if (sparse) {
  // TODO(Ishmael): If (sparse) line 72 in SAG_logistic_BLAS
  }
  /* Counting*/
  for (int i = 0; i < train_set.nSamples; i++) {
    if (train_set.covered[i] != 0) train_set.nCovered++;
  }
  double nCovered = train_set.nCovered;
  for (trainer.iter = 0; trainer.iter < trainer.maxIter; trainer.iter++) {
    // Runing Iteration
    trainer.step(&trainer, &model, &train_set);
  }
  if (sparse) {
    // TODO(Ishmael): Line 153 in SAG_logistic_BLAS
  }

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

