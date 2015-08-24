#ifndef SAG_CONSTANT_H_
#define SAG_CONSTANT_H_
#include "sag_common.h"
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"
/* Interface */
void sag_constant(GlmTrainer * trainer, GlmModel * model, Dataset * dataset);
/* Core */
void _sag_constant(double * w, double * Xt, double * y, double lambda,
                   double alpha, double * d, double * g, loss_grad_fun grad_fun,
                   int * covered, double * nCovered, int nSamples, int nVars,
                   int sparse, int * jc, int * ir,
                   int * lastVisited, double * cumSum, double tol, int maxIter,
                   int monitor, double * monitor_w, int * iter_count,
                   int * convergence_code);
#endif /* SAG_CONSTANT_H_ */
