#ifndef GLM_MODELS_H_
#define GLM_MODELS_H_
#include <math.h>


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

/* Gaussian*/
double gaussian_loss(double y, double innerProd);
double gaussian_grad(double y, double innerProd);
/* Binomial*/
double binomial_loss(double y, double innerProd);
double binomial_grad(double y, double innerProd);
/*  Exponential */
double exponential_loss(double y, double innerProd);
double exponential_grad(double y, double innerProd);
/* Poisson */
double poisson_loss(double y, double innerProd);
double poisson_grad(double y, double innerProd);

#endif /* GLM_MODELS_H_ */
