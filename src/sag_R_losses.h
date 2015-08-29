#ifndef SAG_R_LOSSES_H_
#define SAG_R_LOSSES_H_
#include "sag_common.h"

double eval_R_model_fun(SEXP fun, SEXP env, double y, double innerProd);

double R_loss_wrapper(double y, double innerProd);
double R_loss_grad_wrapper(double y, double innerProd);

#endif /* SAG_R_LOSSES_H_ */
