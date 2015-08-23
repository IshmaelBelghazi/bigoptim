#ifndef TRAINER_H_
#define TRAINER_H_
#include "sag_common.h"
/*============\
| GLM trainer |
\============*/
/* Trainer structs */
typedef struct GlmTrainer {
  double lambda;  // scalar regularization parameter
  double alpha;  // Constant step-size
  double * d;  // Initial Approximation of average Gradient
  double * g;  // Previous derivative of loss
  int iter_count;  // Post training iteration count
  int maxIter;  // Maximum number of iterations
  int stepSizeType;  //  default is 1 to use 1/L, set to 2 to use 2(L
                     //  + n * mu)
  double precision;
  double tol;  // Tolerance

  SAG_TYPE fit_alg;  // SAG fit algorithm
  int monitor;  // Monitor training after every pass
  double * monitor_w;  // Monitoring weigthts
} GlmTrainer;

GlmTrainer make_GlmTrainer(SEXP lambda, SEXP alpha, SEXP d, SEXP g, SEXP maxIter,
                           SEXP stepSizeType, SEXP tol, SEXP fit_alg, SEXP monitor);
#endif /* TRAINER_H_ */

