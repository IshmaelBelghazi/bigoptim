#ifndef ENTRYPOINT_SAG_ADAPTIVE_H_
#define ENTRYPOINT_SAG_ADAPTIVE_H_
#include "sag_common.h"
#include "sag_adaptive.h"
#include "glm_models.h"
#include "dataset.h"
#include "trainers.h"
#include "utils.h"


SEXP C_sag_adaptive(SEXP wInit, SEXP Xt, SEXP y, SEXP lambda,
                    SEXP LmaxInit, SEXP LiInit, SEXP randVals, SEXP dInit,
                    SEXP gInit, SEXP coveredInit, SEXP increasing, SEXP family,
                    SEXP tol, SEXP sparse, SEXP monitor);

#endif /* ENTRYPOINT_SAG_ADAPTIVE_H_ */
