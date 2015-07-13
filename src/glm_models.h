#ifndef GLM_MODELS_H_
#define GLM_MODELS_H_
#include <math.h>


typedef struct {
  double * w; // Weights (p, 1)
  double (*loss)(double, double);
  double (*grad)(double, double);
} GlmModel;

typedef enum {GAUSSIAN, BINOMIAL} GlmType; 

/* Gaussian*/
double gaussian_grad(double y, double innerProd);
double gaussian_loss(double y, double innerProd);

/* Binomial*/
double binomial_grad(double y, double innerProd);
double binomial_loss(double y, double innerProd);

#endif /* GLM_MODELS_H_ */
