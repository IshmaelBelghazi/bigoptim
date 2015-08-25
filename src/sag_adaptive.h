#ifndef SAG_ADAPTIVE_H_
#define SAG_ADAPTIVE_H_
#include "sag_common.h"
#include "glm_models.h"
#include "dataset.h"
#include "trainers.h"

/* Interface */
void sag_adaptive(GlmTrainer *trainer, GlmModel *model, Dataset *dataset);
/* Core */
void _sag_adaptive(double *restrict w, const double *restrict Xt,
                   const double *restrict y, double *restrict Li,
                   double *restrict Lmax, const int increasing, const int nVars,
                   const int nSamples, int *restrict covered,
                   double *restrict unCoveredMatrix, double *restrict LiMatrix,
                   double *restrict nDescendants, double *restrict nCovered,
                   double *restrict Lmean, const int nLevels,
                   const int nextpow2, const int maxIter, const double lambda,
                   double alpha, const double precision, const double tol,
                   const loss_fun loss_function, const loss_grad_fun grad_fun,
                   const int sparse, const int *restrict jc,
                   const int *restrict ir, int *restrict lastVisited,
                   double *restrict cumSum, double *restrict d,
                   double *restrict g, const int monitor,
                   double *restrict monitor_w, int *restrict iter_count,
                   int *restrict convergence_code);
#endif /* SAG_ADAPTIVE_H_ */
