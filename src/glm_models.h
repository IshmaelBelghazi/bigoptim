#ifndef GLM_MODELS_H
#define GLM_MODELS_H
#include <math.h>

typedef struct{
  double (*loss)(double, double);
  double (*grad)(double, double);
} GlmModel;


double logistic_grad(double y, double innerProd);
double logistic_loss(double y, double innerProd);


#endif /* GLM_MODELS_H */
