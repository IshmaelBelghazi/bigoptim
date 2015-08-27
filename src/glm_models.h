#ifndef GLM_MODELS_H_
#define GLM_MODELS_H_
#include "sag_common.h"
#include "sag_C_dynload_posix.h"
/* Avaible Glm models */
typedef enum { GAUSSIAN, BINOMIAL, EXPONENTIAL, POISSON, C_SHARED } GlmType;
/* Model struct */
typedef struct {
  double *w; // Weights (p, 1)
  loss_fun loss;
  loss_grad_fun grad;
  /* Model type */
  GlmType model_type;
  /* Container for shared lib loss function, grad function and handle */
  dyn_fun_container dyn_shlib_container;
} GlmModel;


/* Initializer */
GlmModel make_GlmModel(SEXP w, SEXP family, SEXP ex_model_params);
/* Generics */
double glm_cost(const double *restrict Xt, const double *restrict y,
                const double *restrict w, const double lambda,
                const int nSamples, const int nVars, const loss_fun glm_loss);
void glm_cost_grad(const double *restrict Xt, const double *restrict y,
                   const double *restrict w, const double lambda,
                   const int nSamples, const int nVars,
                   const loss_grad_fun glm_grad, double *restrict grad);
/* Gaussian */
double gaussian_loss(const double y, const double innerProd);
double gaussian_loss_grad(const double y, const double innerProd);
double gaussian_cost(const double *restrict Xt, const double *restrict y,
                     const double *restrict w, const double lambda,
                     const int nSamples, const int nVars);
void gaussian_cost_grad(const double *restrict Xt, const double *restrict y,
                        const double *restrict w, const double lambda,
                        const int nSamples, const int nVars,
                        double *restrict grad);

/* Binomial */
double binomial_loss(const double y, const double innerProd);
double binomial_loss_grad(const double y, const double innerProd);
double binomial_cost(const double *restrict Xt, const double *restrict y,
                     const double *restrict w, const double lambda,
                     const int nSamples, const int nVars);
void binomial_cost_grad(const double *restrict Xt, const double *restrict y,
                        const double *restrict w, const double lambda,
                        const int nSamples, const int nVars,
                        double *restrict grad);

/*  Exponential */
double exponential_loss(const double y, const double innerProd);
double exponential_loss_grad(const double y, const double innerProd);
double exponential_cost(const double *restrict Xt, const double *restrict y,
                        const double *restrict w, const double lambda,
                        const int nSamples, const int nVars);
void exponential_cost_grad(const double *restrict Xt, const double *restrict y,
                           const double *restrict w, const double lambda,
                           const int nSamples, const int nVars,
                           double *restrict grad);

/* Poisson */
double poisson_loss(const double y, const double innerProd);
double poisson_loss_grad(const double y, const double innerProd);
double poisson_cost(const double *restrict Xt, const double *restrict y,
                    const double *restrict w, const double lambda,
                    const int nSamples, const int nVars);
void poisson_cost_grad(const double *restrict Xt, const double *restrict y,
                       const double *restrict w, const double lambda,
                       const int nSamples, const int nVars,
                       double *restrict grad);

#endif /* GLM_MODELS_H_ */
