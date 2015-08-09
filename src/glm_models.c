#include "glm_models.h"
const int one= 1;
/*=========\
| BINOMIAL |
\=========*/
/* loss function */
double binomial_loss(double y, double innerProd) {
  return log(1 + exp(-y * innerProd));
}
/* Gradient of loss function */
double binomial_loss_grad(double y, double innerProd) {
  return -y/(1 + exp(y * innerProd));
}
/* Cost function*/
double binomial_cost(double * Xt, double * y, double * w, double lambda,
                       const int nSamples, const int nVars) {
  double nll = 0;  // Negative log likelihood
  double cost = 0;
  for (int i = 0; i < nSamples; i++) {
    double innerProd = F77_CALL(ddot)(&nVars, w, &one, &Xt[nVars * i], &one);
    nll += binomial_loss(y[i], innerProd);
  }
  cost = nll/(double)nSamples;
  cost += 0.5 * lambda * F77_CALL(ddot)(&nVars, w, &one, w, &one);
  return cost;
}

/* Gradient of cost function*/
void binomial_cost_grad(double * Xt, double * y, double * w, double lambda,
                           const int nSamples, const int nVars, double * grad) {
  for (int i = 0; i < nSamples; i++) {
    double innerProd = F77_CALL(ddot)(&nVars, w, &one, &Xt[nVars * i], &one);
    double innerProd_grad = binomial_loss_grad(y[i], innerProd);
    F77_CALL(daxpy)(&nVars, &innerProd_grad, &Xt[nVars * i], &one, grad, &one);
  }
  // Dividing each entry by nSamples
  double averaging_factor = 1/(double)nSamples;
  F77_CALL(dscal)(&nVars, &averaging_factor, grad, &one);
  // Adding regularization
  F77_CALL(daxpy)(&nVars, &lambda, w, &one, grad, &one);
}

/*=========\
| GAUSSIAN |
\=========*/
/* loss function */
double gaussian_loss(double y, double innerProd) {
  return 0.5 * (innerProd - y) * (innerProd - y);
}
/*Gradient of loss function*/
double gaussian_loss_grad(double y, double innerProd) {
  return innerProd - y;
}

/*============\
| EXPONENTIAL |
\============*/
/* Exponential loss function */
double exponential_loss(double y, double innerProd) {
  return exp(-y * innerProd);
}
/* Exponential gradient function */
double exponential_grad(double y, double innerProd) {
  return -y * exp(-y * innerProd);
}

/*========\
| POISSON |
\========*/
/* Poisson loss function */
double poisson_loss(double y, double innerProd) {
  return exp(innerProd) - y * innerProd;
}
/* Poisson gradient function */
double poisson_loss_grad(double y, double innerProd) {
  return exp(innerProd) - y;
}
