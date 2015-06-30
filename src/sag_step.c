#include "sag_step.h"

const static int one = 1;

void _sag_constant_iteration(GlmTrainer * trainer,
                             GlmModel * model,
                             Dataset * dataset) {

    // TODO(Ishmael): Rename k
  int nVars = dataset->nVars;
  double * w = model->w;
  double * Xt = dataset->Xt;
  double * y = dataset->y;
  double * d = trainer->d;
  double * g = trainer->g;

  /* Select next training example */
  int i = dataset->iVals[trainer->iter] - 1;  // start from 1?
  /* Compute current values of needed parameters */
  if (dataset->sparse && trainer->iter > 0) {
    //TODO(Ishmael): Line 91 in SAG_logistic_BLAS
  }
    
  /* Compute derivative of loss */
  double innerProd = 0;
  if (dataset->sparse) {
    //TODO(Ishmael): Line 104 in SAG_LOGISTIC_BLAS
  } else {
    innerProd = F77_CALL(ddot)(&nVars, w, &one, &Xt[nVars * i], &one);
    
  }

  double grad = model->grad(y[i], innerProd);

  /* Update direction */
  double scaling = 0;
  if (dataset->sparse) {
    // TODO(Ishmael): Line 117 in SAG_logistic_BLAS
  } else {
    scaling = grad - g[i];
    F77_CALL(daxpy)(&nVars, &scaling, &Xt[nVars * i], &one, d, &one);
  }

  /* Store derivative of loss */
  g[i] = grad;
  /* Update the number of examples that we have seen */
  if (dataset->covered[i] == 0) {
    dataset->covered[i] = 1;
    dataset->nCovered++;
  }

  /* Update parameters */
  if (dataset->sparse) {
    // TODO(Ishmael): Line 135 in SAG_logistic_BLAS
  } else {
    scaling = 1 - trainer->alpha * trainer->lambda;
    F77_CALL(dscal)(&nVars, &scaling, w, &one);
    scaling = -trainer->alpha/dataset->nCovered;
    F77_CALL(daxpy)(&nVars, &scaling, d, &one, w, &one);
  }

}
  /* if (sparse) { */
  /*   // TODO(Ishmael): Line 153 in SAG_logistic_BLAS */
  /* } */
