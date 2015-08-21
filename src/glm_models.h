#ifndef GLM_MODELS_H_
#define GLM_MODELS_H_
#include "sag_common.h"

/* Loss functions pointer */
typedef double (*loss_fun)(double, double);
typedef double (*loss_grad_fun)(double, double);

/* Model struct */
typedef struct {
  double * w; // Weights (p, 1)
  loss_fun loss;
  loss_grad_fun grad;
} GlmModel;

typedef enum {GAUSSIAN,
              BINOMIAL,
              EXPONENTIAL,
              POISSON
} GlmType;

/* Gaussian */
double gaussian_loss(double y, double innerProd);
double gaussian_loss_grad(double y, double innerProd);
double gaussian_cost(double * Xt, double * y, double * w,
                     double lambda, int nSamples, int nVars);
void gaussian_cost_grad(double * Xt, double * y, double * w, double lambda,
                        const int nSamples, const int nVars, double * grad);

/* Binomial */
double binomial_loss(double y, double innerProd);
double binomial_loss_grad(double y, double innerProd);
double binomial_cost(double * Xt, double * y, double * w,
                      double lambda, int nSamples, int nVars);
void binomial_cost_grad(double * Xt, double * y, double * w, double lambda,
                         const int nSamples, const int nVars, double * grad);

/*  Exponential */
double exponential_loss(double y, double innerProd);
double exponential_loss_grad(double y, double innerProd);
double exponential_cost(double * Xt, double * y, double * w,
                     double lambda, int nSamples, int nVars);
void exponential_cost_grad(double * Xt, double * y, double * w, double lambda,
                        const int nSamples, const int nVars, double * grad);

/* Poisson */
double poisson_loss(double y, double innerProd);
double poisson_loss_grad(double y, double innerProd);
double poisson_cost(double * Xt, double * y, double * w,
                     double lambda, int nSamples, int nVars);
void poisson_cost_grad(double * Xt, double * y, double * w, double lambda,
                        const int nSamples, const int nVars, double * grad);

#endif /* GLM_MODELS_H_ */
