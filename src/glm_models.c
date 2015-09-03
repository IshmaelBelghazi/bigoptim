#include "glm_models.h"
const static int one = 1;
SEXP R_loss_fun;
SEXP R_loss_fun_env;
SEXP R_grad_fun;
SEXP R_grad_fun_env;
/*=============\
| inititalizer |
\=============*/
GlmModel make_GlmModel(SEXP w, SEXP family, SEXP ex_model_params) {

  GlmModel model = {.w = REAL(w),
                    .model_type = *INTEGER(family)};

  /* Choosing family */
  switch (model.model_type) {
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
  case C_SHARED:
    ;; // Empty statement. Labels can only be followed by statements and
       // declarations are not statements.
    const char * shared_path = STRING_VALUE((getListElement(ex_model_params,
                                                            "lib_file_path")));
    const char * loss_symbol = STRING_VALUE(getListElement(ex_model_params,
                                                                "loss_name"));
    const char * loss_grad_symbol = STRING_VALUE(getListElement(ex_model_params,
                                                                     "grad_name"));
    model.dyn_shlib_container = load_C_shared_model(shared_path,
                                                    loss_symbol,
                                                    loss_grad_symbol);
    model.loss = model.dyn_shlib_container.dyn_loss_fun;
    model.grad = model.dyn_shlib_container.dyn_loss_grad_fun;
    break;
  case R:
    /* Assigning R functions and environement to global variables */
    
    R_loss_fun = getListElement(ex_model_params, "R_loss_fun");
    R_loss_fun_env = getListElement(ex_model_params, "R_loss_fun_env");
    R_grad_fun = getListElement(ex_model_params, "R_grad_fun");
    R_grad_fun_env = getListElement(ex_model_params, "R_grad_fun_env");
    /* assigning R losses wrappers */
    model.loss = R_loss_wrapper;
    model.grad = R_loss_grad_wrapper;
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
  double innerProd = 0;
  #pragma omp parallel for private(innerProd) reduction(+ : nll)
  for (int i = 0; i < nSamples; i++) {
    innerProd = F77_CALL(ddot)(&nVars, w, &one, &Xt[nVars * i], &one);
    nll += (*glm_loss)(y[i], innerProd);
  }
  cost = nll / (double)nSamples;
  cost += 0.5 * lambda * F77_CALL(ddot)(&nVars, w, &one, w, &one);
  return cost;
}
/* Generic glm grad function */
void glm_cost_grad(const double *restrict Xt, const double *restrict y,
                   const double *restrict w, const double lambda,
                   const int nSamples, const int nVars,
                   const loss_grad_fun glm_grad, double *restrict grad) {
  for (int i = 0; i < nSamples; i++) {
    double innerProd = F77_CALL(ddot)(&nVars, w, &one, &Xt[nVars * i], &one);
    double innerProd_grad = (*glm_grad)(y[i], innerProd);
    F77_CALL(daxpy)(&nVars, &innerProd_grad, &Xt[nVars * i], &one, grad, &one);
  }
  /* Dividing each entry by nSamples */
  double averaging_factor = 1 / (double)nSamples;
  F77_CALL(dscal)(&nVars, &averaging_factor, grad, &one);
  /* Adding regularization */
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
