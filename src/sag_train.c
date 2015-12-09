#include "sag_train.h"


/* Trainer */

void train(GlmTrainer* trainer, GlmModel* model, Dataset* dataset) {
  SAG_TYPE fit_alg = trainer->fit_alg;
  /* Counting previously seen examples */
  switch (fit_alg) {
  case CONSTANT:
    count_covered_samples(dataset, 0);
    sag_constant(trainer, model, dataset);
    break;
  case LINESEARCH:
    count_covered_samples(dataset, 0);
    sag_linesearch(trainer, model, dataset);
    break;
  case ADAPTIVE:
    count_covered_samples(dataset, 1);
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
