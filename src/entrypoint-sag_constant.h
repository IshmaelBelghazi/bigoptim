#ifndef ENTRYPOINT_SAG_CONSTANT_H_
#define ENTRYPOINT_SAG_CONSTANT_H_
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "utils.h"
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"

SEXP C_sag_constant(SEXP w, SEXP Xt, SEXP y, SEXP lambda,
                    SEXP stepSize, SEXP iVals, SEXP d, SEXP g,
                    SEXP covered, SEXP family, SEXP tol, SEXP sparse);


#endif /* ENTRYPOINT_SAG_CONSTANT_H_ */