#ifndef SAG_LINESEARCH_H_
#define SAG_LINESEARCH_H_
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "utils.h"
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"

/* Interface */
void sag_linesearch(GlmTrainer * trainer, GlmModel * model, Dataset * dataset);
/* Core */
void _sag_linesearch(double * w, double * Xt, double * y, double lambda,
                     double alpha, int stepSizeType, double precision, double * Li,
                     double * d, double * g, loss_fun loss_function, loss_grad_fun grad_function,
                     int * iVals, int * covered, double * nCovered, int nVars, int nSamples,
                     int sparse, int * ir, int * jc, int * lastVisited,
                     double * cumSum, int maxIter, double tol);
#endif /* SAG_LINESEARCH_H_ */
