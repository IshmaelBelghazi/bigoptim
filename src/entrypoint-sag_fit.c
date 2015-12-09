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
               SEXP family,
               SEXP fit_alg,
               SEXP ex_model_params,  // Parameters for external model functions
                                      // (external C shared, dynamically
                                      // compiled, and R callbacks)
               SEXP sparse,
               SEXP monitor) {
  /*===============\
  | Error Checking |
  \===============*/
  validate_inputs(wInit, Xt, y, dInit, gInit, coveredInit, sparse);
  /* Initializing protection counter */
  int nprot = 0;
  /* Duplicating objects to be modified */
  SEXP w = PROTECT(duplicate(wInit)); nprot++;
  SEXP d = PROTECT(duplicate(dInit)); nprot++;
  SEXP g = PROTECT(duplicate(gInit)); nprot++;
  SEXP covered = PROTECT(duplicate(coveredInit)); nprot++;
  SEXP Li = PROTECT(duplicate(LiInit)); nprot++;
  SEXP Lmax = PROTECT(duplicate(LmaxInit)); nprot++;
  /* Initializing monitor weights data structure */
  SEXP monitor_w = PROTECT(initialize_monitor(monitor, maxiter, Xt));
  if (*INTEGER(monitor)) nprot++;  // Nothing to Protect if no monitoring
  /*======\
  | Input |
  \======*/
  /* Initializing dataset */
  Dataset train_set = make_Dataset(Xt, y, covered, Lmax, Li, increasing, fit_alg, sparse);
  /* Initializing Trainer */
  GlmTrainer trainer = make_GlmTrainer(lambda, alpha, d, g,
                                       maxiter, stepSizeType, tol, fit_alg,
                                       monitor, monitor_w);
  /* Initializing Model */
  GlmModel model = make_GlmModel(w, family, ex_model_params);
  /*============================\
  | Stochastic Average Gradient |
  \============================*/
  /* Training */
  train(&trainer, &model, &train_set);
  /* Clean up */
  cleanup(&trainer, &model, &train_set);
  /*=======\
  | Return |
  \=======*/
  SEXP convergence_code = PROTECT(allocVector(INTSXP, 1)); nprot++;
  *INTEGER(convergence_code) = trainer.convergence_code;
  SEXP iter_count = PROTECT(allocVector(INTSXP, 1)); nprot++;
  *INTEGER(iter_count) = trainer.iter_count;
  /* Assigning variables to SEXP list */
  SEXP results = PROTECT(allocVector(VECSXP, 9)); nprot++;
  INC_APPLY(SEXP, SET_VECTOR_ELT, results, w, d, g, covered, Li, Lmax, convergence_code, iter_count, monitor_w); // in utils.h
  /* Creating SEXP for list names */
  SEXP results_names = PROTECT(allocVector(STRSXP, 9)); nprot++;
  INC_APPLY_SUB(char *, SET_STRING_ELT, mkChar, results_names, "w", "d", "g",
                "covered", "Li", "Lmax", "convergence_code", "iter_count", "monitor_w");
  setAttrib(results, R_NamesSymbol, results_names);
  /* ---------------------------------------------------------------------------*/
  UNPROTECT(nprot);
  return results;
}
