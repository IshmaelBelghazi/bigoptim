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
  return 0.5 * (innerProd - y) * (innerProd - y);
}
/*Gradient of loss function*/
double gaussian_grad(double y, double innerProd) {
  return innerProd - y;
}

/*============\
| Exponential |
\============*/
/* Exponential loss function */
double exponential_loss(double y, double innerProd) {
  return exp(-y * innerProd);
}
/* Exponential gradient function */
double exponential_grad(double y, double innerProd) {
  return -y * exp(-y * innerProd);
}

/*========\
| Poisson |
\========*/
/* Poisson loss function */
double poisson_loss(double y, double innerProd) {
  return exp(innerProd) - y * innerProd;
}
/* Poisson gradient function */
double poisson_grad(double y, double innerProd) {
  return exp(innerProd) - y;

}
