#ifndef DATASET_H_
#define DATASET_H_

typedef struct {
  double * Xt;  // Transposed features (p, n)
  double * y;   // Targets (n, *)
  int * iVals;  // Sequence of example to choose from
  double * randVals;  // (maxiter, 2) - sequence of random values for
                      // the algorithm to use
  int * covered;  // Whether the example has been visited (n, 1)
  double nCovered;  // Number of Covered Examples
  int nSamples;  // Number of examples
  int nVars;  // number of variables
  int sparse;  // Are the matrices sparse?
  /* Lipschitz Constants */
  double * Lmax;  // Initial approximation of global Lipschitz
                  // constant
  double * Li;  // Initial Approximation of individual Lipschitz
                // constant
  double Lmean;  //Mean of Lipschitz coefficient of covered varibles
  int increasing;  // 1 to allow lipschitz constant to increase. 0 To
                   // only allow them to decrease
  
  /* Adaptive data-structure */
  int nextpow2;  // Next power of 2
  int nLevels;  // Number of levels
  double * nDescendants;  // Number of descendents in the tree
  double * unCoveredMatrix;  // Number of descendents that are still
                             // uncovered
  double * LiMatrix;  // Sums of Lipschitz constant of loss over descendents
  
} Dataset;

#endif /* DATASET_H_*/
