#include "sag_step.h"

const static int one = 1;

/*============================\
| SAG with constant step size |
\============================*/

void _sag_constant_iteration(GlmTrainer * trainer,
                             GlmModel * model,
                             Dataset * dataset) {

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


/*====================\
| SAG with linesearch |
\====================*/
void _sag_linesearch_iteration(GlmTrainer * trainer,
                               GlmModel * model,
                               Dataset * dataset) {

  int nVars = dataset->nVars;
  double * w = model->w;
  double * Xt = dataset->Xt;
  double * y = dataset->y;
  double * d = trainer->d;
  double * g = trainer->g;
  double * Li = trainer-> Li;
  
  /* Select next training example */
  int i = dataset->iVals[trainer->iter] - 1;
  if (dataset->sparse && trainer->iter > 0) {
    //TODO(Ishmael): SAGlineSearch_logistic_BLAS.c line 119
  }
  /* Compute derivative of loss */
  double innerProd = 0;
  if (dataset->sparse) {
    // TODO(Ishmael): SAGlineSearch_logistic_BLAS.c line 132
  } else {
    innerProd = F77_CALL(ddot)(&nVars, w, &one, &Xt[nVars * i], &one);
  }

  double grad = model->grad(y[i], innerProd);
  
  /* Update Direction */
  double scaling = 0;
  if (dataset->sparse) {
    // TODO(Ishmael): SAGlineSearch_logistic_BLAS.c line 144
  } else {
    scaling = grad - g[i];
    F77_CALL(daxpy)(&nVars, &scaling, &Xt[i * nVars], &one, d, &one);
  }
  /* Store Derivatives of loss */
  g[i] = grad;
  /* Update the number of examples that we have seen */
  if (dataset->covered[i] == 0) {
    dataset->covered[i] = 1; dataset->nCovered++;
  }

  /* Line-search for Li */
  double fi = model->loss(y[i], innerProd);
  /* Compute f_new as the function value obtained by taking 
   * a step size of 1/Li in the gradient direction */
  double wtx = innerProd;
  double xtx = F77_CALL(ddot)(&dataset->nVars, &Xt[i * dataset->nVars], &one, &Xt[i * dataset->nVars], &one);
  double gg = grad * grad * xtx;
  innerProd = wtx - xtx * grad/(*Li);
  
  double fi_new = model->loss(y[i], innerProd);
  while (gg > trainer->precision && fi_new > fi - gg/(2 * (*Li))) {
    *Li *= 2;
    innerProd = wtx - xtx * grad/(*Li);
    fi_new = log(1 + exp(-y[i] * innerProd));
  }
    
  /* Compute step size */
  if (trainer->stepSizeType == 1) {
    trainer->alpha = 1/(*Li + trainer->lambda);
  } else {
    trainer->alpha = 2/(*Li + (dataset->nSamples + 1) * trainer->lambda);       
  }
  /* Update Parameters */
  if (dataset->sparse) {
    // TODO(Ishmael):  SAGlineSearch_logistic_BLAS.c line 187
  } else {
    scaling = 1 - trainer->alpha * trainer->lambda;
    F77_CALL(dscal)(&nVars, &scaling, w, &one);
    scaling = -(trainer->alpha)/(dataset->nCovered);
    F77_CALL(daxpy)(&nVars, &scaling, d, &one, w, &one);
  }

  /* Decrease value of Lipschitz constant */
  *Li *= pow(2.0, -1.0/dataset->nSamples); 
  
}

  /* if (sparse) { */
  /*   // TODO(Ishmael):  SAGlineSearch_logistic_BLAS.c line 208 */
  /* } */

/*===========================\
| SAG with Adaptive Sampling |
\===========================*/
