#ifndef GLM_MODELS_H
#define GLM_MODELS_H
#include <math.h>


double logistic_grad(double y, double innerProd);
double logistic_loss(double y, double innerProd);

#endif /* GLM_MODELS_H */
