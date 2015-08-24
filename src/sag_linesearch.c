#include "sag_linesearch.h"

const static int one = 1;

void sag_linesearch(GlmTrainer *trainer, GlmModel *model, Dataset *dataset) {

  /* Unpacking Structs */

  /* Dataset */
  double *y = dataset->y;
  double *Xt = dataset->Xt;
  double *Li = dataset->Li;

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
  double precision = trainer->precision;
  int stepSizeType = trainer->stepSizeType;
  double tol = trainer->tol;
  /* Model */
  double *w = model->w;
  loss_fun loss_function = model->loss;
  loss_grad_fun grad_function = model->grad;

  /* Sparse related variables */
  int sparse = dataset->sparse;
  int *jc = NULL, *ir = NULL;
  int *lastVisited = NULL;
  double *cumSum = NULL;
  if (sparse) {
    /* Sparse indices*/
    jc = dataset->jc;
    ir = dataset->ir;
    /* Allocate Memory Needed for lazy update */
    cumSum = Calloc(maxIter, double);
    lastVisited = Calloc(nVars, int);
  }

  /* Approximate gradients*/
  double *g = trainer->g;
  double *d = trainer->d;

  /* Monitoring */
  int monitor = trainer->monitor;
  double * monitor_w = trainer->monitor_w;

  /* Training */
  _sag_linesearch(w, Xt, y, lambda,
                  alpha, stepSizeType, precision, Li,
                  d, g, loss_function, grad_function,
                  covered, nCovered, nVars, nSamples,
                  sparse, ir, jc, lastVisited,
                  cumSum, maxIter, tol, monitor, monitor_w);


  /* Deallocating */
  if (sparse) {
  Free(lastVisited);
  Free(cumSum);
  }
}

  void _sag_linesearch(double * w, double * Xt, double * y, double lambda,
                       double alpha, int stepSizeType, double precision, double * Li,
                       double * d, double * g, loss_fun loss_function, loss_grad_fun grad_function,
                       int * covered, double * nCovered, int nVars, int nSamples,
                       int sparse, int * ir, int * jc, int * lastVisited,
                       double * cumSum, int maxIter, double tol, int monitor, double * monitor_w) {

    GetRNGstate();
  /* Training variables*/
  int i = 0;
  double c = 1.0;
  double scaling = 0, innerProd = 0, grad = 0;
  double fi = 0, fi_new = 0;
  double gg = 0, wtx = 0, xtx = 0;

  double agrad_norm = R_PosInf;

  int stop_condition = 0;
  /* Training Loop */
  int k = 0;  // TODO(Ishmael): Consider using the register keyword
  /* Monitoring */
  int pass_num = 0; // For weights monitoring
  if ( monitor  && k % nSamples == 0 ) {
    if (DEBUG) {
      R_TRACE("effective pass # %d. saving weights.", pass_num);
    }
    F77_CALL(dcopy)(&nVars, w, &one, &monitor_w[nVars * pass_num], &one);
  }
  while (!stop_condition){
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
        innerProd += w[ir[j]] * Xt[j];
      }
      innerProd *= c;
    } else {
      innerProd = F77_CALL(ddot)(&nVars, w, &one, &Xt[nVars * i], &one);
    }

    grad = grad_function(y[i], innerProd);

    /* Update direction */
    if (sparse) {
      for (int j = jc[i]; j < jc[i + 1]; j++) {
        d[ir[j]] += Xt[j] * (grad - g[i]);
      }
    } else {
      scaling = grad - g[i];
      F77_CALL(daxpy)(&nVars, &scaling, &Xt[nVars * i], &one, d, &one);
    }

    /* Store derivative of loss */
    g[i] = grad;

    /* Update the number of examples that we have seen */
    if (covered[i] == 0) {
      covered[i] = 1;
      (*nCovered)++;
    }

    /* Line-search for Li */
    fi = loss_function(y[i], innerProd);
    /* Compute f_new as the function value obtained by taking
     * a step size of 1/Li in the gradient direction */
    wtx = innerProd;
    if (sparse) {
      xtx = 0;
      for (int j = jc[i]; j < jc[i + 1]; j++) {
        xtx += Xt[j] * Xt[j];
      }
    } else {
      xtx = F77_CALL(ddot)(&nVars, &Xt[nVars * i], &one, &Xt[nVars * i], &one);
    }
    gg = grad * grad * xtx;
    innerProd = wtx - xtx * grad / (*Li);
    fi_new = loss_function(y[i], innerProd);
    /*printf("fi = %e, fi_new = %e, gg = %e\n",fi,fi_new,gg);*/
    while (gg > precision && fi_new > fi - gg / (2 * (*Li))) {
      /*printf("Lipschitz Backtracking (k = %d, fi = %e, fi_new = %e, 1/Li =
       * %e)\n",k+1,fi,fi_new,1/(*Li));*/
      *Li *= 2;
      innerProd = wtx - xtx * grad / (*Li);
      fi_new = loss_function(y[i], innerProd);
    }

    /* Compute step size */
    if (stepSizeType == 1) {
      alpha = 1 / (*Li + lambda);
    } else {
      alpha = 2 / (*Li + (nSamples + 1) * lambda);
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

    /* Decrease value of Lipschitz constant */
    *Li *= pow(2.0, -1.0 / nSamples);

    /* if (k % nSamples == 0 && DEBUG) { */
    /*   R_TRACE("pass %d: cost=%f", k/nSamples, binomial_cost(Xt, y, w, lambda, nSamples, nVars)); */
    /* } */
    /* Incrementing iteration count */
    k++;
    /* Checking Stopping criterions */
    if (!sparse) {
    agrad_norm = get_cost_agrad_norm(w, d, lambda, *nCovered, nSamples, nVars);
    }
    stop_condition = (k >= maxIter) || (agrad_norm <= tol);
    /* Monitoring */
    if ( monitor && k % nSamples == 0) {
      pass_num++;
      if (DEBUG) {
        R_TRACE("effective pass # %d. saving weights.", pass_num);
      }
      F77_CALL(dcopy)(&nVars, w, &one, &monitor_w[nVars * pass_num], &one);
    }
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
  PutRNGstate();
  R_TRACE("Final approxite gradient norm: %F", agrad_norm);
}
