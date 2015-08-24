#include "entrypoint-sag_fit.h"

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
SEXP C_sag_fit(SEXP wInit, SEXP Xt, SEXP y, SEXP lambda,
               SEXP alpha,  // SAG Constant Step size
               SEXP stepSizeType, // SAG Linesearch
               SEXP LiInit,  // SAG Linesearch and Adaptive
               SEXP LmaxInit,  // SAG Adaptive
               SEXP increasing,  // SAG Adaptive
               SEXP dInit, SEXP gInit, SEXP coveredInit,
               SEXP tol, SEXP maxiter,
               SEXP family, SEXP fit_alg,
               SEXP sparse, SEXP monitor) {
  /*===============\
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
  GlmTrainer trainer = make_GlmTrainer(lambda, alpha, d, g, maxiter,
                                       stepSizeType, tol, fit_alg, monitor);
  R_TRACE("Trainer initialized");
  /* Monitoring weights */
  R_TRACE("Setting up monitor");
  int n_passes = trainer.maxIter / train_set.nSamples;
  SEXP monitor_w;
  if (trainer.monitor) {
    monitor_w = PROTECT(allocMatrix(REALSXP, train_set.nVars, n_passes + 1)); nprot++;
    Memzero(REAL(monitor_w), (n_passes + 1) * train_set.nVars);
    trainer.monitor_w = REAL(monitor_w);
  } else {
    monitor_w = R_NilValue;
  }
  R_TRACE("Monitor set");
  /* Initializing Model */
  R_TRACE("Initializing model");
  GlmModel model = make_GlmModel(w, family);
  R_TRACE("Model initialized");
  /*============================\
  | Stochastic Average Gradient |
  \============================*/
  /* Counting previously covered example for manual warm starting */
  /* Training */
  R_TRACE("Training ...");
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
  SEXP results = PROTECT(allocVector(VECSXP, 8)); nprot++;
  INC_APPLY(SEXP, SET_VECTOR_ELT, results, w, d, g, covered, Li, convergence_code, iter_count, monitor_w); // in utils.h
  /* Creating SEXP for list names */
  SEXP results_names = PROTECT(allocVector(STRSXP, 8)); nprot++;
  INC_APPLY_SUB(char *, SET_STRING_ELT, mkChar, results_names, "w", "d", "g",
                "covered", "Li", "convergence_code", "iter_count", "monitor_w");
  setAttrib(results, R_NamesSymbol, results_names);
  R_TRACE("Return S-EXP all set up");
  /* ---------------------------------------------------------------------------*/
  UNPROTECT(nprot);
  return results;
}