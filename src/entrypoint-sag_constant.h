#ifndef ENTRYPOINT_SAG_CONSTANT_H_
#define ENTRYPOINT_SAG_CONSTANT_H_
#include "sag_common.h"
#include "sag_constant.h"
#include "utils.h"
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"

SEXP C_sag_constant(SEXP wInit, SEXP Xt, SEXP y, SEXP lambda,
                    SEXP stepSize, SEXP iVals, SEXP dInit, SEXP gInit,
                    SEXP coveredInit, SEXP family, SEXP tol, SEXP sparse,
                    SEXP monitor);


#endif /* ENTRYPOINT_SAG_CONSTANT_H_ */
