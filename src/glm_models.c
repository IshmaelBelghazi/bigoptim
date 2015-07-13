#include "glm_models.h"

/*=========\
| BINOMIAL |
\=========*/
/* loss function (not error function)*/
double binomial_loss(double y, double innerProd) {
  return log(1 + exp(-y * innerProd));
}
/*Gradient of loss function*/
double binomial_grad(double y, double innerProd) {
  return -y/(1 + exp(y * innerProd));
}
/*=========\
| GAUSSIAN |
\=========*/
/* loss function (not error function)*/
double gaussian_loss(double y, double innerProd) {
  return 0.5 * pow(innerProd - y, 2);
}
/*Gradient of loss function*/
double gaussian_grad(double y, double innerProd) {
  return innerProd - y;
}

