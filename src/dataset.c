#include "dataset.h"

/* Initializer */
Dataset make_Dataset(SEXP Xt, SEXP y, SEXP covered, SEXP Lmax,
                     SEXP Li, SEXP increasing, SEXP fit_alg, SEXP sparse) {

  SAG_TYPE alg = *INTEGER(fit_alg);
  Dataset data_set = {.y = REAL(y),
                      .covered = INTEGER(covered),
                      .nCovered = 0,
                      .sparse = *INTEGER(sparse)};

  switch (alg) {
  case CONSTANT:
    break;
  case LINESEARCH:
    data_set.Li = REAL(Li);
    break;
  case ADAPTIVE:
    data_set.Li = REAL(Li);
    data_set.Lmax = REAL(Lmax);
    data_set.increasing = *INTEGER(increasing);
    break;
  default:
    error("Unrecognied fit algorithm");
    break;
  }

  /* Initializing sample*/
  CHM_SP cXt;
  if (data_set.sparse) {
    cXt = AS_CHM_SP(Xt);
    /* Sparse Array pointers*/
    data_set.ir = cXt->i;
    data_set.jc = cXt->p;
    data_set.Xt = cXt->x;
    /* Sparse Variables */
    data_set.nVars = cXt->nrow;
    data_set.nSamples = cXt->ncol;
  } else {
    data_set.Xt = REAL(Xt);
    data_set.nSamples = INTEGER(GET_DIM(Xt))[1];
    data_set.nVars = INTEGER(GET_DIM(Xt))[0];
  }
  return data_set;
}

/* Utils */
/* Counts Covered samples*/
void count_covered_samples(Dataset* dataset, int compute_covered_mean) {
  dataset->nCovered = 0;
  dataset->Lmean = 0;
  if (DEBUG) R_TRACE("nsamples: %d, nVars: %d", dataset->nSamples, dataset->nVars);
  for (int i = 0; i < dataset->nSamples; i++) {
    if (dataset->covered[i] != 0) {
      dataset->nCovered++;
      if (compute_covered_mean) dataset->Lmean++;
    }
  }
  if (dataset->nCovered > 0 && compute_covered_mean) {
    dataset->Lmean /= dataset->nCovered;
  }
}
