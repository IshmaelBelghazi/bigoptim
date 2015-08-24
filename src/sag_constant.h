#ifndef SAG_CONSTANT_H_
#define SAG_CONSTANT_H_
#include "sag_common.h"
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"
/* Interface */
void sag_constant(GlmTrainer *trainer, GlmModel *model, Dataset *dataset);
/* Core */
void _sag_constant(double *restrict w, const double *restrict Xt,
                   const double *y, const double lambda, const double alpha,
                   double *restrict d, double *restrict g,
                   const loss_grad_fun grad_fun, int *restrict covered,
                   double *restrict nCovered, const int nSamples,
                   const int nVars, const int sparse, const int *restrict jc,
                   const int *restrict ir, int *restrict lastVisited,
                   double *restrict cumSum, const double tol, const int maxIter,
                   const int monitor, double *restrict monitor_w,
                   int *restrict iter_count, int *restrict convergence_code);
#endif /* SAG_CONSTANT_H_ */
