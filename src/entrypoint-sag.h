#ifndef ENTRYPOINT_SAG_H_
#define ENTRYPOINT_SAG_H_
#include "sag_common.h"
#include "dataset.h"
#include "glm_models.h"
#include "trainers.h"
#include "sag_warm.h"

SEXP C_sag(SEXP wInit, SEXP Xt, SEXP y, SEXP lambdas,
           SEXP alpha,  // SAG Constant Step size
           SEXP stepSizeType, // SAG Linesearch
           SEXP LiInit,  // SAG Linesearch and Adaptive
           SEXP LmaxInit,  // SAG Adaptive
           SEXP increasing,  // SAG Adaptive
           SEXP dInit, SEXP gInit, SEXP coveredInit,
           SEXP tol, SEXP maxiter,
           SEXP family,
           SEXP fit_alg,
           SEXP sparse);

#endif  /* ENTRYPOINT_SAG_H_*/
