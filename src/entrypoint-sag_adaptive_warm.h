#ifndef ENTRYPOINT_SAG_ADAPTIVE_WARM_H_
#define ENTRYPOINT_SAG_ADAPTIVE_WARM_H_
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "sag_adaptive_warm.h"
#include "utils.h"
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"
#include "Matrix.h"
#include "cholmod.h"

SEXP C_sag_adaptive_warm(SEXP wInit, SEXP Xt, SEXP y, SEXP lambdas,
                         SEXP LmaxInit, SEXP LiInit, SEXP randVals, SEXP dInit,
                         SEXP gInit, SEXP coveredInit, SEXP increasing,
                         SEXP family, SEXP tol, SEXP sparse);

#endif /* ENTRYPOINT_SAG_ADAPTIVE_WARM_H_ */
