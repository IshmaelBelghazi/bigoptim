#ifndef TRAINER_H_
#define TRAINER_H_
#include "dataset.h"
#include "glm_models.h"

typedef struct GlmTrainer {
  double lambda; // scalar regularization parameter
  double alpha;  // Constant step-size
  double * d;  // Initial Approximation of average Gradient
  double * g;  // Previous derivative of loss
  int iter;  // Current Iteration count
  int maxIter;  // Maximum number of iterations
  /* Performs a single step of the attached SAG algorithm */
  void (*step)(struct GlmTrainer *, GlmModel *, Dataset *); 
} GlmTrainer;

#endif /* TRAINER_H_ */
