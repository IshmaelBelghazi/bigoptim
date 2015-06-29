#ifndef TRAINER_H_
#define TRAINER_H_

typedef struct {
  double * w; // Weights (p, 1)
  double lambda; // scalar regularization parameter
  double alpha;  // Constant step-size
  double * d;  // Initial Approximation of average Gradient
  double * g;  // Previous derivative of loss
  int maxIter;  // Maximum number of iterations
  int * iVals;  // Sequence of example to choose from
} SAGConstant;

#endif /* TRAINER_H_ */
