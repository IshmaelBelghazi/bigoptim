#include "sag_train.h"


/* Trainer */

void train(GlmTrainer* trainer, GlmModel* model, Dataset* dataset) {
  if (DEBUG) R_TRACE("Starting Training");
  SAG_TYPE fit_alg = trainer->fit_alg;
  if (DEBUG) R_TRACE("fit_alg=%d", fit_alg);
  if (DEBUG) R_TRACE("Dispathching fit algorithm");
  /* Counting previously seen examples */
  switch (fit_alg) {
  case CONSTANT:
    if (DEBUG) R_TRACE("Counting seen sample points");
    count_covered_samples(dataset, 0);
    if (DEBUG) R_TRACE("Seen sample points counted");
    sag_constant(trainer, model, dataset);
    break;
  case LINESEARCH:
    if (DEBUG) R_TRACE("Counting seen sample points");
    count_covered_samples(dataset, 0);
    if (DEBUG) R_TRACE("Seen sample points counted");
    sag_linesearch(trainer, model, dataset);
    break;
  case ADAPTIVE:
    if (DEBUG) R_TRACE("adaptive selected");
    if (DEBUG) R_TRACE("Counting seen sample points and computing covered meam");
    count_covered_samples(dataset, 1);
    if (DEBUG) R_TRACE("Seen sample points counted amd covered mean computed");
    sag_adaptive(trainer, model, dataset);
  default:
    break;
  }
  if (DEBUG) R_TRACE("Training Completed");
}
/* Return List */
/* Make return list */
SEXP make_return_list(GlmTrainer* trainer, GlmModel* model, Dataset* dataset) {

}
/* Cleanup */
void cleanup(GlmTrainer* trainer, GlmModel* model, Dataset* dataset) {
  /* Dynamic loading cleanup */
  if (model->model_type == C_SHARED) {
    if (!model->dyn_shlib_container.handle) {
      dlclose(model->dyn_shlib_container.handle);
    }
  }
}
