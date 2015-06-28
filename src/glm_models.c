#include "glm_models.h"

/*=======\
| Losses |
\=======*/
double logistic_loss(double y, double innerProd) {
  return log(1 + exp(-y * innerProd));
}

/*==========\
| Gradients |
\==========*/
double logistic_grad(double y, double innerProd) {
  return -y/(1 + exp(y * innerProd));
}

