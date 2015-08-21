#ifndef ENTRYPOINT_SAG_CONSTANT_WARM_H_
#define ENTRYPOINT_SAG_CONSTANT_WARM_H_
#include "sag_common.h"
#include "sag_constant_warm.h"
#include "utils.h"
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"

SEXP C_sag_constant_warm(SEXP wInit, SEXP Xt, SEXP y, SEXP lambdas,
                         SEXP stepSize, SEXP iVals, SEXP dInit, SEXP gInit,
                         SEXP coveredInit, SEXP family, SEXP tol, SEXP sparse);


#endif /* ENTRYPOINT_SAG_CONSTANT_WARM_H_ */
