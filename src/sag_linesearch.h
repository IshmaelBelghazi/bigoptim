#ifndef SAG_LINESEARCH_H_
#define SAG_LINESEARCH_H_
#include "sag_common.h"
#include "utils.h"
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"

/* Interface */
void sag_linesearch(GlmTrainer * trainer, GlmModel * model, Dataset * dataset);
/* Core */
void _sag_linesearch(double *restrict w, const double *restrict Xt,
                     const double *restrict y, const double lambda,
                     double alpha, const int stepSizeType,
                     const double precision, double *restrict Li,
                     double *restrict d, double *restrict g,
                     const loss_fun loss_function, const loss_grad_fun grad_function,
                     int *restrict covered, double *restrict nCovered,
                     const int nVars, const int nSamples, const int sparse,
                     const int *restrict ir, const int *restrict jc,
                     int *restrict lastVisited, double *restrict cumSum,
                     const int maxIter, const double tol, const int monitor,
                     double *restrict monitor_w, int *restrict iter_count,
                     int *restrict convergence_code);
#endif /* SAG_LINESEARCH_H_ */
