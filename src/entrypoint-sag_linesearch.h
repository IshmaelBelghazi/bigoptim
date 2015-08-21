#ifndef ENTRYPOINT_SAG_LINESEARCH_H_
#define ENTRYPOINT_SAG_LINESEARCH_H_
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "sag_linesearch.h"
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"
#include "utils.h"
#include "Matrix.h"
#include "cholmod.h"

SEXP C_sag_linesearch(SEXP wInit, SEXP Xt, SEXP y, SEXP lambda,
                      SEXP stepSizeInit, SEXP iVals, SEXP dInit, SEXP gInit,
                      SEXP coveredInit, SEXP stepSizeType, SEXP family, SEXP tol,
                      SEXP sparse, SEXP monitor);


#endif /* ENTRYPOINT_SAG_LINESEARCH_H_ */
