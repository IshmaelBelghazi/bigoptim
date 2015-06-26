#include "glm_models.h"

/*==========\
| Gradients |
\==========*/
double logistic_grad(double y, double innerProd) {
  return -y/(1 + exp(y * innerProd));
}
