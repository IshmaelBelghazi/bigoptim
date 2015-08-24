#include "glm_models.h"

const int one = 1;

/*=============\
| inititalizer |
\=============*/

GlmModel make_GlmModel(SEXP w, SEXP family) {

  GlmModel model = {.w = REAL(w)};
  GlmType model_type = *INTEGER(family);

  /* Choosing family */
  switch (model_type) {
  case GAUSSIAN:
    model.loss = gaussian_loss;
    model.grad = gaussian_loss_grad;
    break;
  case BINOMIAL:
    model.loss = binomial_loss;
    model.grad = binomial_loss_grad;
    break;
  case EXPONENTIAL:
    model.loss = exponential_loss;
    model.grad = exponential_loss_grad;
    break;
  case POISSON:
    model.loss = poisson_loss;
    model.grad = poisson_loss_grad;
    break;
  default:
    error("Unrecognized glm family");
    break;
  }

  return model;
}

/*=========\
| GENERICS |
\=========*/

/* Generic glm cost functions */
double glm_cost(const double *restrict Xt, const double *restrict y,
                const double *restrict w, const double lambda,
                const int nSamples, const int nVars, const loss_fun glm_loss) {

  double nll = 0; // Negative log likelihood
  double cost = 0;
  // #pragma omp parallel for reduction(+ : nll)
  for (int i = 0; i < nSamples; i++) {
    double innerProd = F77_CALL(ddot)(&nVars, w, &one, &Xt[nVars * i], &one);
    nll += (*glm_loss)(y[i], innerProd);
  }
  cost = nll / (double)nSamples;
  cost += 0.5 * lambda * F77_CALL(ddot)(&nVars, w, &one, w, &one);
  return cost;
}
/* Generic glm grad function */
void glm_cost_grad(const double *restrict Xt, const double *restrict y,
                   const double *restrict w, const double lambda,
                   const int nSamples, const int nVars, const loss_grad_fun glm_grad,
                   double *restrict grad) {
  // #pragma omp parallel for
  for (int i = 0; i < nSamples; i++) {
    double innerProd = F77_CALL(ddot)(&nVars, w, &one, &Xt[nVars * i], &one);
    double innerProd_grad = (*glm_grad)(y[i], innerProd);
    F77_CALL(daxpy)(&nVars, &innerProd_grad, &Xt[nVars * i], &one, grad, &one);
  }
  // Dividing each entry by nSamples
  double averaging_factor = 1 / (double)nSamples;
  F77_CALL(dscal)(&nVars, &averaging_factor, grad, &one);
  // Adding regularization
  F77_CALL(daxpy)(&nVars, &lambda, w, &one, grad, &one);
}

/*=========\
| BINOMIAL |
\=========*/

/* loss function */
double binomial_loss(const double y, const double innerProd) {
  return log1p(exp(-y * innerProd));
}

/* Gradient of loss function */
double binomial_loss_grad(const double y, const double innerProd) {
  return -y / (1 + exp(y * innerProd));
}

/* Cost function*/
double binomial_cost(const double *restrict Xt, const double *restrict y,
                     const double *restrict w, const double lambda,
                     const int nSamples, const int nVars) {

  return glm_cost(Xt, y, w, lambda, nSamples, nVars, &binomial_loss);
}

/* Gradient of cost function*/
void binomial_cost_grad(const double *restrict Xt, const double *restrict y,
                        const double *restrict w, const double lambda,
                        const int nSamples, const int nVars,
                        double *restrict grad) {
  glm_cost_grad(Xt, y, w, lambda, nSamples, nVars, &binomial_loss_grad, grad);
}

/*=========\
| GAUSSIAN |
\=========*/

/* loss function */
double gaussian_loss(const double y, const double innerProd) {
  return 0.5 * (innerProd - y) * (innerProd - y);
}

/*Gradient of loss function*/
double gaussian_loss_grad(const double y, const double innerProd) {
  return innerProd - y;
}

/* Cost function*/
double gaussian_cost(const double *restrict Xt, const double *restrict y,
                     const double *restrict w, const double lambda,
                     const int nSamples, const int nVars) {
  return glm_cost(Xt, y, w, lambda, nSamples, nVars, &gaussian_loss);
}

/* Gradient of cost function*/
void gaussian_cost_grad(const double *restrict Xt, const double *restrict y,
                        const double *restrict w, const double lambda,
                        const int nSamples, const int nVars,
                        double *restrict grad) {
  glm_cost_grad(Xt, y, w, lambda, nSamples, nVars, &gaussian_loss_grad, grad);
}

/*============\
| EXPONENTIAL |
\============*/

/* Exponential loss function */
double exponential_loss(const double y, const double innerProd) {
  return exp(-y * innerProd);
}
/* Exponential gradient function */
double exponential_loss_grad(const double y, const double innerProd) {
  return -y * exp(-y * innerProd);
}
/* Cost function*/
double exponential_cost(const double *restrict Xt, const double *restrict y,
                        const double *w, const double lambda,
                        const int nSamples, const int nVars) {
  return glm_cost(Xt, y, w, lambda, nSamples, nVars, &exponential_loss);
}
/* Gradient of cost function*/
void exponential_cost_grad(const double *restrict Xt, const double *restrict y,
                           const double *restrict w, const double lambda,
                           const int nSamples, const int nVars,
                           double *restrict grad) {
  glm_cost_grad(Xt, y, w, lambda, nSamples, nVars, &exponential_loss_grad,
                grad);
}
/*========\
| POISSON |
\========*/
/* Poisson loss function */
double poisson_loss(const double y, const double innerProd) {
  return exp(innerProd) - y * innerProd;
}
/* Poisson gradient function */
double poisson_loss_grad(const double y, const double innerProd) {
  return exp(innerProd) - y;
}
/* Cost function*/
double poisson_cost(const double *restrict Xt, const double *restrict y,
                    const double *restrict w, const double lambda,
                    const int nSamples, const int nVars) {
  return glm_cost(Xt, y, w, lambda, nSamples, nVars, &poisson_loss);
}
/* Gradient of cost function*/
void poisson_cost_grad(const double *restrict Xt, const double *restrict y,
                       const double *restrict w, const double lambda,
                       const int nSamples, const int nVars,
                       double *restrict grad) {
  glm_cost_grad(Xt, y, w, lambda, nSamples, nVars, &poisson_loss_grad, grad);
}
