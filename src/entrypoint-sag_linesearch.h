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

SEXP C_sag_linesearch(SEXP w, SEXP Xt, SEXP y, SEXP lambda,
                      SEXP stepSize, SEXP iVals, SEXP d, SEXP g,
                      SEXP covered, SEXP stepSizeType, SEXP family, SEXP tol,
                      SEXP sparse);


#endif /* ENTRYPOINT_SAG_LINESEARCH_H_ */
