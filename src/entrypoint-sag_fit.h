#ifndef ENTRYPOINT_SAG_FIT_H_
#define ENTRYPOINT_SAG_FIT_H_
#include "sag_common.h"
#include "sag_train.h"
#include "dataset.h"
#include "glm_models.h"
#include "trainers.h"
#include "utils.h"

SEXP C_sag_fit(SEXP wInit, SEXP Xt, SEXP y, SEXP lambda,
               SEXP alpha,  // SAG Constant Step size
               SEXP stepSizeType, // SAG Linesearch
               SEXP LiInit,  // SAG Linesearch and Adaptive
               SEXP LmaxInit,  // SAG Adaptive
               SEXP increasing,  // SAG Adaptive
               SEXP dInit, SEXP gInit, SEXP coveredInit,
               SEXP tol, SEXP maxiter,
               SEXP family, SEXP fit_alg,
               SEXP sparse, SEXP monitor);

#endif /* ENTRYPOINT_SAG_FIT_H_ */
