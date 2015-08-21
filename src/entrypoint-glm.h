#ifndef SAG_GLM_ENTRY_POINT_H_
#define SAG_GLM_ENTRY_POINT_H_
#include "sag_common.h"
#include "glm_models.h"

/*=========\
| BINOMIAL |
\=========*/
SEXP C_binomial_cost(SEXP Xt, SEXP y, SEXP w, SEXP lambda);
SEXP C_binomial_cost_grad(SEXP Xt, SEXP y, SEXP w, SEXP lambda);
/*=========\
| GAUSSIAN |
\=========*/
SEXP C_gaussian_cost(SEXP Xt, SEXP y, SEXP w, SEXP lambda);
SEXP C_gaussian_cost_grad(SEXP Xt, SEXP y, SEXP w, SEXP lambda);
/*============\
| EXPONENTIAL |
\============*/
SEXP C_exponential_cost(SEXP Xt, SEXP y, SEXP w, SEXP lambda);
SEXP C_exponential_cost_grad(SEXP Xt, SEXP y, SEXP w, SEXP lambda);
/*========\
| POISSON |
\========*/
SEXP C_poisson_cost(SEXP Xt, SEXP y, SEXP w, SEXP lambda);
SEXP C_poisson_cost_grad(SEXP Xt, SEXP y, SEXP w, SEXP lambda);

#endif /* SAG_GLM_ENTRY_POINT_H_ */
