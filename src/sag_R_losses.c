#include "sag_R_losses.h"
SEXP R_loss_fun;
SEXP R_loss_fun_env;
SEXP R_grad_fun;
SEXP R_grad_fun_env;

/* Evaluates R function fun in env at parameters y and innerProd */
double eval_R_model_fun(SEXP fun, SEXP env, double y, double innerProd) {
  SEXP output = PROTECT(output = eval(lang3(fun,
                                            ScalarReal(y),
                                            ScalarReal(innerProd)),
                                      env));
  UNPROTECT(1);
  return asReal(output);
}

double R_loss_wrapper(double y, double innerProd) {
  return eval_R_model_fun(R_loss_fun, R_loss_fun_env, y, innerProd);
}

double R_loss_grad_wrapper(double y, double innerProd) {
  return eval_R_model_fun(R_grad_fun, R_grad_fun_env, y, innerProd);
}
