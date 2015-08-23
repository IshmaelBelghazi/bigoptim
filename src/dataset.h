#ifndef DATASET_H_
#define DATASET_H_

#include "sag_common.h"

typedef struct {
  double * Xt;  // Transposed features (p, n)
  double * y;   // Targets (n, 1)

  int * covered;  // Whether the example has been visited (n, 1)
  double nCovered;  // Number of Covered Examples
  int nSamples;  // Number of examples
  int nVars;  // number of variables
  int sparse;  // Are the matrices sparse?

  /* Lipschitz Constants */
  double * Lmax;  // Initial approximation of global Lipschitz constant
  double * Li;  // Initial Approximation of individual Lipschitz constant
  double Lmean;  // Mean of Lipschitz coefficient of covered varibles
  int increasing;  // 1 to allow lipschitz constant to increase. 0 To 1 only allow them to decrease
  /* Sparse indices */
  int * ir;
  int * jc;

} Dataset;

/* Initializer */
Dataset make_Dataset(SEXP Xt, SEXP y, SEXP covered, SEXP Lmax,
                     SEXP Li, SEXP increasing, SEXP fit_alg, SEXP sparse);
/* Utils */
void count_covered_samples(Dataset* dataset, int compute_covered_mean);
#endif /* DATASET_H_*/

