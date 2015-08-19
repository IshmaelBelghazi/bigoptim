#ifndef ENTRYPOINT_SAG_ADAPTIVE_H_
#define ENTRYPOINT_SAG_ADAPTIVE_H_
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "glm_models.h"
#include "dataset.h"
#include "trainers.h"
#include "utils.h"
#include "Matrix.h"
#include "cholmod.h"


SEXP C_sag_adaptive(SEXP w, SEXP Xt, SEXP y, SEXP lambda, SEXP Lmax,
                    SEXP Li, SEXP randVals, SEXP d, SEXP g, SEXP covered,
                    SEXP increasing, SEXP family, SEXP tol, SEXP sparse);

#endif /* ENTRYPOINT_SAG_ADAPTIVE_H_ */
