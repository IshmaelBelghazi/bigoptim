#ifndef TRAINER_H_
#define TRAINER_H_
#include "dataset.h"
#include "glm_models.h"
//TODO(Ishmael): Lmax and Li depends on the dataset. They should be
//moved to Dataset
typedef struct GlmTrainer {
  double lambda; // scalar regularization parameter
  double alpha;  // Constant step-size
  double * d;  // Initial Approximation of average Gradient
  double * g;  // Previous derivative of loss
  int iter;  // Current Iteration count
  int maxIter;  // Maximum number of iterations
  int stepSizeType;  //  default is 1 to use 1/L, set to 2 to use 2(L
                     //  + n * mu)
  double * Lmax;  // Initial approximation of global Lipschitz
                  // constant
  double * Li;  // Initial Approximation of individual Lipschitz
                // constant
  double precision;
  /* Performs a single step of the attached SAG algorithm */
  void (*step)(struct GlmTrainer *, GlmModel *, Dataset *); 
} GlmTrainer;

#endif /* TRAINER_H_ */
