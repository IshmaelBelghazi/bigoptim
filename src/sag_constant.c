#include "sag_constant.h"

const static int one = 1;
void sag_constant(GlmTrainer *trainer, GlmModel *model, Dataset *dataset) {

  /* Unpacking Structs */
  /* Dataset */
  double *y = dataset->y;
  double *Xt = dataset->Xt;
  /* Dimensions */
  int nVars = dataset->nVars;
  int nSamples = dataset->nSamples;
  /* Sampling */
  int *covered = dataset->covered;
  double *nCovered = &dataset->nCovered;
  /* Training parameters */
  int maxIter = trainer->maxIter;
  double lambda = trainer->lambda;
  double alpha = trainer->alpha;
  double tol = trainer->tol;
  /* Model */
  double *w = model->w;
  loss_grad_fun grad_fun = model->grad;
  /* Sparse related variables */
  int sparse = dataset->sparse;
  int *jc = NULL, *ir = NULL;
  int *lastVisited = NULL;
  double *cumSum = NULL;
  if (sparse) {
    if (DEBUG) R_TRACE("Populating sparse pointers");
    /* Sparce indices*/
    jc = dataset->jc;
    ir = dataset->ir;
    /* Allocate Memory Needed for lazy update */
    if (DEBUG) R_TRACE("Allocating sparse variables");
    cumSum = Calloc(maxIter, double);
    lastVisited = Calloc(nVars, int);
  }
  /* Approximate gradients*/
  double *g = trainer->g;
  double *d = trainer->d;
  /* Monitoring */
  int monitor = trainer->monitor;
  double *monitor_w = trainer->monitor_w;
  /* Convergence diagnostic */
  int * iter_count = &trainer->iter_count;
  int * convergence_code = &trainer->convergence_code;
  /* Training */
  _sag_constant(w, Xt, y, lambda, alpha, d, g, grad_fun, covered, nCovered,
                nSamples, nVars, sparse, jc, ir, lastVisited, cumSum, tol,
                maxIter, monitor, monitor_w, iter_count, convergence_code);
  /* Deallocating */
  if (sparse) {
    Free(lastVisited);
    Free(cumSum);
  }
}

void _sag_constant(double * restrict w, const double * restrict Xt, const double *y, const double lambda,
                   const double alpha, double * restrict d, double *restrict g, const loss_grad_fun grad_fun,
                   int * restrict covered, double * restrict nCovered, const int nSamples, const int nVars,
                   const int sparse, const int * restrict jc, const int * restrict ir, int *restrict lastVisited,
                   double * restrict cumSum, const double tol, const int maxIter, const int monitor,
                   double * restrict monitor_w, int * restrict iter_count, int * restrict convergence_code) {

  GetRNGstate();
  /* Training variables*/
  register int i = 0;
  double c = 1.0;
  double scaling = 0;
  double  innerProd = 0;
  double grad = 0;
  double agrad_norm = R_PosInf;
  int stop_condition = 0;
  /* Training Loop */
  register int k = 0; // TODO(Ishmael): Consider using the register keyword
  // Monitoring
  int pass_num = 0; // For weights monitoring
  if (monitor && k % nSamples == 0) {
    if (DEBUG) {
      if (DEBUG) R_TRACE("effective pass # %d. saving weights.", pass_num);
    }
    F77_CALL(dcopy)(&nVars, w, &one, &monitor_w[nVars * pass_num], &one);
  }
  while (!stop_condition) {
    /* Select next training example */
    i = (int)floor(runif(0, 1) * nSamples);
    /* Compute current values of needed parameters */
    if (sparse && k > 0) {
      for (int j = jc[i]; j < jc[i + 1]; j++) {
        if (lastVisited[ir[j]] == 0) {
          w[ir[j]] -= d[ir[j]] * cumSum[k - 1];
        } else {
          w[ir[j]] -=
              d[ir[j]] * (cumSum[k - 1] - cumSum[lastVisited[ir[j]] - 1]);
        }
        lastVisited[ir[j]] = k;
      }
    }

    /* Compute derivative of loss */
    if (sparse) {
      innerProd = 0;
      for (int j = jc[i]; j < jc[i + 1]; j++) {
        // if (DEBUG) R_TRACE("jc[%d]=%f, jc[%d]=%f",i, jc[i], i + 1, jc[i + 1]);
        innerProd += w[ir[j]] * Xt[j];
      }
      innerProd *= c;
    } else {
      innerProd = F77_CALL(ddot)(&nVars, w, &one, &Xt[nVars * i], &one);
    }

    grad = grad_fun(y[i], innerProd);

    /* Update direction */
    if (sparse) {
      for (int j = jc[i]; j < jc[i + 1]; j++) {
        d[ir[j]] += Xt[j] * (grad - g[i]);
      }
    } else {
      scaling = grad - g[i];
      F77_CALL(daxpy)(&nVars, &scaling, &Xt[i * nVars], &one, d, &one);
    }

    /* Store derivative of loss */
    g[i] = grad;
    /* Update the number of examples that we have seen */
    if (covered[i] == 0) {
      covered[i] = 1;
      (*nCovered)++;
    }

    /* Update parameters */
    if (sparse) {
      c *= 1 - alpha * lambda;
      if (k == 0) {
        cumSum[0] = alpha / (c * *nCovered);
      } else {
        cumSum[k] = cumSum[k - 1] + alpha / (c * *nCovered);
      }
    } else {
      scaling = 1 - alpha * lambda;
      F77_CALL(dscal)(&nVars, &scaling, w, &one);
      scaling = -alpha / *nCovered;
      F77_CALL(daxpy)(&nVars, &scaling, d, &one, w, &one);
    }

    /* if (k % nSamples == 0 && DEBUG) { */
    /*     if (DEBUG) R_TRACE("pass %d: cost=%f", k/nSamples, binomial_cost(Xt, y, w,
     * lambda, nSamples, nVars)); */
    /*       } */
    /* Incrementing iteration count */
    k++;
    /* Checking Stopping criterions */
    if (!sparse) {
      if (tol > 0) agrad_norm = get_cost_agrad_norm(w, d, lambda, *nCovered, nSamples, nVars);
    }
    stop_condition = (k >= maxIter) || (agrad_norm <= tol);
    /* Monitoring */
    if (monitor && k % nSamples == 0) {
      pass_num++;
      if (DEBUG) {
        if (DEBUG) R_TRACE("effective pass # %d. saving weights.", pass_num);
      }
      F77_CALL(dcopy)(&nVars, w, &one, &monitor_w[nVars * pass_num], &one);
    }
  }
  PutRNGstate();
  if (DEBUG) R_TRACE("Final approxite gradient norm: %F", agrad_norm);
  /* Setting final iteration count */
  *iter_count = k;
  /* Checking Convergence condition */
  if (agrad_norm < tol) {
    *convergence_code = 1;
  } else {
    *convergence_code = 0;
    if (tol > 0) warning("Optimisation stopped before convergence. Try incrasing maximum number of iterations");
  }

  if (sparse) {
    for (int j = 0; j < nVars; j++) {
      if (lastVisited[j] == 0) {
        w[j] -= d[j] * cumSum[maxIter - 1];
      } else {
        w[j] -= d[j] * (cumSum[maxIter - 1] - cumSum[lastVisited[j] - 1]);
      }
    }
    scaling = c;
    F77_CALL(dscal)(&nVars, &scaling, w, &one);
  }

}
