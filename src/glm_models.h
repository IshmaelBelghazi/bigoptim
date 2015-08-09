#ifndef GLM_MODELS_H_
#define GLM_MODELS_H_
#include <stdio.h>
#include <math.h>
#include <R.h>
#include <R_ext/BLAS.h>

typedef struct {
  double * w; // Weights (p, 1)
  double (*loss)(double, double);
  double (*grad)(double, double);
} GlmModel;

typedef enum {GAUSSIAN,
              BINOMIAL,
              EXPONENTIAL,
              POISSON
} GlmType;

/* Gaussian */
double gaussian_loss(double y, double innerProd);
double gaussian_loss_grad(double y, double innerProd);
/* Binomial */
double binomial_loss(double y, double innerProd);
double binomial_loss_grad(double y, double innerProd);
double binomial_cost(double * Xt, double * y, double * w,
                      double lambda, int nSamples, int nVars);
void binomial_cost_grad(double * Xt, double * y, double * w, double lambda,
                         const int nSamples, const int nVars, double * grad);
/*  Exponential */
double exponential_loss(double y, double innerProd);
double exponential_grad(double y, double innerProd);
/* Poisson */
double poisson_loss(double y, double innerProd);
double poisson_loss_grad(double y, double innerProd);

#endif /* GLM_MODELS_H_ */
