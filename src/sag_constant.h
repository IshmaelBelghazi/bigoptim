#ifndef SAG_CONSTANT_H_
#define SAG_CONSTANT_H_
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "utils.h"
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"

/* Interface */
void sag_constant(GlmTrainer * trainer, GlmModel * model, Dataset * dataset);
/* Core */
void _sag_constant(double * w, double * Xt, double * y, double lambda, double alpha,
                   double * d, double * g, loss_grad_fun grad_fun,
                   int * iVals, int * covered, double * nCovered,
                   int nSamples, int nVars, int sparse, int * jc, int * ir,
                   int * lastVisited, double * cumSum, double tol, int maxIter,
                   int monitor, double * monitor_w);
#endif /* SAG_CONSTANT_H_ */
