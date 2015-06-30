#ifndef DATASET_H_
#define DATASET_H_

typedef struct {
  double * Xt;  // Transposed features (p, n)
  double * y;   // Targets (n, *)
  int * iVals;  // Sequence of example to choose from
  int * covered;  // Whether the example has been visited (n, 1)
  double nCovered;  // Number of Covered Examples
  int nSamples;  // Number of examples
  int nVars;  // number of variables
  int sparse;  // Are the matrices sparse?
  
} Dataset;

#endif /* DATASET_H_*/
