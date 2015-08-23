#include "sag_train.h"


/* Trainer */

void train(GlmTrainer* trainer, GlmModel* model, Dataset* dataset) {
  R_TRACE("Starting Training");
  SAG_TYPE fit_alg = trainer->fit_alg;
  R_TRACE("fit_alg=%d", fit_alg);
  R_TRACE("Dispathching fit algorithm");
  /* Counting previously seen examples */
  switch (fit_alg) {
  case CONSTANT:
    R_TRACE("Counting seen sample points");
    count_covered_samples(dataset, 0);
    R_TRACE("Seen sample points counted");
    sag_constant(trainer, model, dataset);
    break;
  case LINESEARCH:
    R_TRACE("Counting seen sample points");
    count_covered_samples(dataset, 0);
    R_TRACE("Seen sample points counted");
    sag_linesearch(trainer, model, dataset);
    break;
  case ADAPTIVE:
    R_TRACE("adaptive selected");
    R_TRACE("Counting seen sample points and computing covered meam");
    count_covered_samples(dataset, 1);
    R_TRACE("Seen sample points counted amd covered mean computed");
    sag_adaptive(trainer, model, dataset);
  default:
    break;
  }
  R_TRACE("Training Completed");
}
