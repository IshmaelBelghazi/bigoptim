#ifndef SAG_ADAPTIVE_H_
#define SAG_ADAPTIVE_H_
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "glm_models.h"
#include "dataset.h"
#include "trainers.h"
#include "utils.h"

/* Interface */
void sag_adaptive(GlmTrainer *trainer, GlmModel *model, Dataset *dataset);
/* Core */
void _sag_adaptive(double *w, double *Xt, double *y, double *Li, double *Lmax,
                   int increasing, int nVars, int nSamples, double *randVals,
                   int *covered, double *unCoveredMatrix, double *LiMatrix,
                   double *nDescendants, double *nCovered, double *Lmean,
                   int nLevels, int nextpow2, int maxIter, double lambda,
                   double alpha, double precision, double tol,
                   loss_fun loss_function, loss_grad_fun grad_fun, int sparse,
                   int *jc, int *ir, int *lastVisited, double *cumSum,
                   double *d, double *g, int monitor, double * monitor_w);
#endif /* SAG_ADAPTIVE_H_ */
