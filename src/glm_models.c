#include "glm_models.h"

/*=======\
| Losses |
\=======*/
/* loss function (not error function)*/
double logistic_loss(double y, double innerProd) {
  return log(1 + exp(-y * innerProd));
}

/*==========\
| Gradients |
\==========*/
/*Gradient of loss function*/
double logistic_grad(double y, double innerProd) {
  return -y/(1 + exp(y * innerProd));
}


