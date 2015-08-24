#include "trainers.h"

/* Trainer Initializer */
GlmTrainer make_GlmTrainer(SEXP lambda, SEXP alpha, SEXP d, SEXP g, SEXP maxIter,
                           SEXP stepSizeType, SEXP tol, SEXP fit_alg, SEXP monitor,
                           SEXP monitor_w) {

  SAG_TYPE alg = *INTEGER(fit_alg);
  if (DEBUG) R_TRACE("fit_alg=%d", alg);
  GlmTrainer trainer = { .lambda = IS_R_NULL(lambda) ? 0 :*REAL(lambda),
                         .d = REAL(d),
                         .g = REAL(g),
                         .maxIter = *INTEGER(maxIter),
                         .tol = *REAL(tol),
                         .iter_count = 0,  // No iterations yet
                         .convergence_code = -1,  // -1 for untrained,
                         .monitor = IS_R_NULL(monitor)? 0 :*INTEGER(monitor),
                         .fit_alg = alg};

  switch(alg) {
  case CONSTANT:
    trainer.alpha = *REAL(alpha);
    break;
  case LINESEARCH:
    trainer.stepSizeType = *INTEGER(stepSizeType);
    break;
  case ADAPTIVE:
    break;
  default:
    error("unrecognized fit algorithm");
    break;
  }
  /* Monitor */
  if (trainer.monitor && !IS_R_NULL(monitor_w)) {
    trainer.monitor_w = REAL(monitor_w);
  }
  return trainer;

}
