#include <stdio.h>
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "utils.h"
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"
#include "sag_step.h"

/* Constant */
const static int sparse = 0;

SEXP C_sag_glm(SEXP w, SEXP Xt, SEXP y, SEXP lambda, SEXP stepSize, SEXP iVals, SEXP randVals,
               SEXP d, SEXP g, SEXP covered, SEXP sag_type) {
  
  // Initializing garbage collection protection counter
  int nprot = 0;
  
  
  Dataset train_set;
  GlmTrainer trainer;
  GlmModel model;
  
  /* Initializing trainer */
  Sag_type type = *INTEGER(sag_type);
  switch (type) {
    case CONSTANT:
      break;
    case LINESEARCH:
      break;
    case ADAPTIVE:
      break;
    default:
      error("Unrecognized algorithm");
  }

  return NULL;

}
