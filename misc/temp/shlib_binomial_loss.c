#include "Rmath.h"

double shared_loss(double y, double innerProd) {
  return log1p(exp(-y * innerProd));
}

double shared_loss_grad(double y, double innerProd) {
  return -y/(1 + exp(y * innerProd));
}
