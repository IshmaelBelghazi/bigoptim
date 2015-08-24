#include "sag_warm.h"

const static int one = 1;

void sag_warm(GlmTrainer* trainer, GlmModel* model, Dataset* dataset,
              double * lambdas, int nLambdas, double * lambda_w) {

  int nVars = dataset->nVars;
  for (int i = 0; i < nLambdas; i++) {
    R_TRACE("Processing lambda[%d]=%f. %d/%d", i, lambdas[i], i + 1, nLambdas);
    /* selecting lambda */
    trainer->lambda = lambdas[i];
    if (DEBUG) {
      R_TRACE("Warm training lambda[%d]=%f", i, lambdas[i]);
    }
    /* Training */
    train(trainer, model, dataset);
    /* Copying weights */
    F77_CALL(dcopy)(&nVars, model->w, &one, &lambda_w[nVars * i], &one);
  }
}
