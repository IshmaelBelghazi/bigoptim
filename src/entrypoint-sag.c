#include "entrypoint-sag.h"

/**
 *     Logistic regression stochastic average gradient trainer
 *
 *     @param w(p, 1) weights
 *     @param Xt(p, n) real feature matrix
 *     @param y(n, 1) {-1, 1} target matrix
 *     @param lambda scalar regularization parameters
 *     @param Li scalar constant step size
 *     @param iVals(max_iter, 1) sequence of examples to choose
 *     @param d(p, 1) initial approximation of average gradient
 *     @param g(n, 1) previous derivatives of loss
 *     @param covered(n, 1) whether the example has been visited
 *     @param stepSizeType scalar default is 1 to use 1/L, set to 2 to
 *     use 2/(L + n*myu)
 *     @return optimal weights (p, 1)
 */

SEXP C_sag(SEXP wInit, SEXP Xt, SEXP y, SEXP lambdas,
           SEXP alpha,  // SAG Constant Step size
           SEXP stepSizeType, // SAG Linesearch
           SEXP LiInit,  // SAG Linesearch and Adaptive
           SEXP LmaxInit,  // SAG Adaptive
           SEXP increasing,  // SAG Adaptive
           SEXP dInit, SEXP gInit, SEXP coveredInit,
           SEXP tol, SEXP maxiter,
           SEXP family, SEXP fit_alg,
           SEXP sparse) {

 /*=============== \
 | Error Checking |
 \===============*/
  R_TRACE("validating inputs");
  validate_inputs(wInit, Xt, y, dInit, gInit, coveredInit, sparse);
  R_TRACE("inputs validated");
  /* Initializing protection counter */
  int nprot = 0;
  /* Duplicating objects to be modified */
  SEXP w = PROTECT(duplicate(wInit)); nprot++;
  SEXP d = PROTECT(duplicate(dInit)); nprot++;
  SEXP g = PROTECT(duplicate(gInit)); nprot++;
  SEXP covered = PROTECT(duplicate(coveredInit)); nprot++;
  SEXP Li = PROTECT(duplicate(LiInit)); nprot++;
  SEXP Lmax = PROTECT(duplicate(LmaxInit)); nprot++;


  /*======\
  | Input |
  \======*/
  /* Initializing dataset */
  R_TRACE("Initializing dataset");
  Dataset train_set = make_Dataset(Xt, y, covered, Lmax, Li, increasing, fit_alg, sparse);
  R_TRACE("Dataset initialized");
  /* Initializing Trainer */
  R_TRACE("Initializing Trainer");
  GlmTrainer trainer = make_GlmTrainer(R_NilValue, alpha, d, g, maxiter,
                                       stepSizeType, tol, fit_alg, R_NilValue);
  R_TRACE("Trainer initialized");
  /* Initializing Model */
  R_TRACE("Initializing model");
  GlmModel model = make_GlmModel(w, family);
  R_TRACE("Model initialized");
  /*============================\
  | Stochastic Average Gradient |
  \============================*/
  /* Initializing lambda/weights Matrix*/
  SEXP lambda_w = PROTECT(allocMatrix(REALSXP, LENGTH(lambdas), train_set.nVars)); nprot++;
  Memzero(REAL(lambda_w), LENGTH(lambdas) * train_set.nVars);
  /* Training */
  R_TRACE("Training ...");
  sag_warm(&trainer, &model, &train_set,
           REAL(lambdas), LENGTH(lambdas), REAL(lambda_w));
  train(&trainer, &model, &train_set);
  R_TRACE("... Training finished");
  /*=======\
  | Return |
  \=======*/
  R_TRACE("Setting up return S-EXP");
  SEXP convergence_code = PROTECT(allocVector(INTSXP, 1)); nprot++;
  *INTEGER(convergence_code) = -1;
  SEXP iter_count = PROTECT(allocVector(INTSXP, 1)); nprot++;
  *INTEGER(iter_count) = trainer.iter_count;
  /* Assigning variables to SEXP list */
  SEXP results = PROTECT(allocVector(VECSXP, 7)); nprot++;
  INC_APPLY(SEXP, SET_VECTOR_ELT, results, lambda_w, d, g, covered, Li, convergence_code, iter_count); // in utils.h
  /* Creating SEXP for list names */
  SEXP results_names = PROTECT(allocVector(STRSXP, 7)); nprot++;
  INC_APPLY_SUB(char *, SET_STRING_ELT, mkChar, results_names, "lambda_w", "d", "g",
                "covered", "Li", "convergence_code", "iter_count");
  setAttrib(results, R_NamesSymbol, results_names);
  R_TRACE("Return S-EXP all set up");
  /* ---------------------------------------------------------------------------*/
  UNPROTECT(nprot);
  return results;
}
