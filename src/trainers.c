#include "trainers.h"

/* Trainer Initializer */
GlmTrainer make_GlmTrainer(SEXP lambda, SEXP alpha, SEXP d, SEXP g, SEXP maxIter,
                           SEXP stepSizeType, SEXP tol, SEXP fit_alg, SEXP monitor) {

  SAG_TYPE alg = *INTEGER(fit_alg);
  R_TRACE("fit_alg=%d", alg);
  GlmTrainer trainer = { .lambda = (lambda == R_NilValue) ? 0:*REAL(lambda),
                         .d = REAL(d),
                         .g = REAL(g),
                         .maxIter = *INTEGER(maxIter),
                         .tol = *REAL(tol),
                         .iter_count = 0,
                         .monitor = (monitor == R_NilValue)? 0 :*INTEGER(monitor),
                         .fit_alg = alg};

  if (alg == CONSTANT) {
    trainer.alpha = *REAL(alpha);
  } else if (alg == LINESEARCH) {
    trainer.stepSizeType = *INTEGER(stepSizeType);
  }

  return trainer;

}
