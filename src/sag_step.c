#include "sag_step.h"

const static int one = 1;
const static int DEBUG = 0;
const static int sparse = 1;
/*============================\
| SAG with constant step size |
\============================*/

void _sag_constant_iteration(GlmTrainer * trainer,
                             GlmModel * model,
                             Dataset * dataset) {

  int nVars = dataset->nVars;
  double * w = model->w;
  double * Xt = dataset->Xt;
  double * y = dataset->y;
  double * d = trainer->d;
  double * g = trainer->g;

  /* Select next training example */

  //if(trainer->iter == 10) error("STOP!");  // Hammer time!
  int i = dataset->iVals[trainer->iter] - 1;  // start from 1?
  /* Compute current values of needed parameters */
  if (dataset->sparse && trainer->iter > 0) {
    //TODO(Ishmael): Line 91 in SAG_logistic_BLAS
  }

  /* Compute derivative of loss */
  double innerProd = 0;
  if (dataset->sparse) {
    //TODO(Ishmael): Line 104 in SAG_LOGISTIC_BLAS
  } else {
    innerProd = F77_CALL(ddot)(&nVars, w, &one, &Xt[nVars * i], &one);
  }

  double grad = model->grad(y[i], innerProd);

  /* Update direction */
  double scaling = 0;
  if (dataset->sparse) {
    // TODO(Ishmael): Line 117 in SAG_logistic_BLAS
  } else {
    scaling = grad - g[i];
    F77_CALL(daxpy)(&nVars, &scaling, &Xt[nVars * i], &one, d, &one);
  }
  /* Store derivative of loss */
  g[i] = grad;
  /* Update the number of examples that we have seen */
  if (dataset->covered[i] == 0) {
    dataset->covered[i] = 1;
    dataset->nCovered++;
  }

  /* Update parameters */
  if (dataset->sparse) {
    // TODO(Ishmael): Line 135 in SAG_logistic_BLAS
  } else {
    scaling = 1 - trainer->alpha * trainer->lambda;
    F77_CALL(dscal)(&nVars, &scaling, w, &one);
    scaling = -trainer->alpha/dataset->nCovered;
    F77_CALL(daxpy)(&nVars, &scaling, d, &one, w, &one);
  }
}
  /* if (sparse) { */
  /*   // TODO(Ishmael): Line 153 in SAG_logistic_BLAS */
  /* } */

/*====================\
| SAG with linesearch |
\====================*/
void _sag_linesearch_iteration(GlmTrainer * trainer,
                               GlmModel * model,
                               Dataset * dataset) {

  int nVars = dataset->nVars;
  double * w = model->w;
  double * Xt = dataset->Xt;
  double * y = dataset->y;
  double * d = trainer->d;
  double * g = trainer->g;
  double * Li = dataset-> Li;

  /* Select next training example */
  int i = dataset->iVals[trainer->iter] - 1;
  if (dataset->sparse && trainer->iter > 0) {
    //TODO(Ishmael): SAGlineSearch_logistic_BLAS.c line 119
  }
  /* Compute derivative of loss */
  double innerProd = 0;
  if (dataset->sparse) {
    // TODO(Ishmael): SAGlineSearch_logistic_BLAS.c line 132
  } else {
    innerProd = F77_CALL(ddot)(&nVars, w, &one, &Xt[nVars * i], &one);
  }

  double grad = model->grad(y[i], innerProd);

  /* Update Direction */
  double scaling = 0;
  if (dataset->sparse) {
    // TODO(Ishmael): SAGlineSearch_logistic_BLAS.c line 144
  } else {
    scaling = grad - g[i];
    F77_CALL(daxpy)(&nVars, &scaling, &Xt[i * nVars], &one, d, &one);
  }
  /* Store derivative of loss */
  g[i] = grad;
  /* Update the number of examples that we have seen */
  if (dataset->covered[i] == 0) {
    dataset->covered[i] = 1; dataset->nCovered++;
  }

  /* Line-search for Li */
  double fi = model->loss(y[i], innerProd);
  /* Compute f_new as the function value obtained by taking
   * a step size of 1/Li in the gradient direction */
  double wtx = innerProd;
  double xtx = F77_CALL(ddot)(&dataset->nVars, &Xt[i * dataset->nVars], &one, &Xt[i * dataset->nVars], &one);
  double gg = grad * grad * xtx;
  innerProd = wtx - xtx * grad/(*Li);

  double fi_new = model->loss(y[i], innerProd);
  while ((gg > trainer->precision) && (fi_new > (fi - gg/(2 * (*Li))))) {
    *Li *= 2;
    innerProd = wtx - xtx * grad/(*Li);
    fi_new = model->loss(y[i], innerProd);
  }

  /* Compute step size */
  if (trainer->stepSizeType == 1) {
    trainer->alpha = 1/(*Li + trainer->lambda);
  } else {
    trainer->alpha = 2/(*Li + (dataset->nSamples + 1) * trainer->lambda);
  }
  /* Update Parameters */
  if (dataset->sparse) {
    // TODO(Ishmael):  SAGlineSearch_logistic_BLAS.c line 187
  } else {
    scaling = 1 - trainer->alpha * trainer->lambda;
    F77_CALL(dscal)(&nVars, &scaling, w, &one);
    scaling = -(trainer->alpha)/(dataset->nCovered);
    F77_CALL(daxpy)(&nVars, &scaling, d, &one, w, &one);
  }

  /* Decrease value of Lipschitz constant */
  *Li *= pow(2.0, -1.0/dataset->nSamples);
}

  /* if (sparse) { */
  /*   // TODO(Ishmael):  SAGlineSearch_logistic_BLAS.c line 208 */
  /* } */

/*===========================\
| SAG with Adaptive Sampling |
\===========================*/

void _sag_adaptive_iteration(GlmTrainer * trainer,
                             GlmModel * model,
                             Dataset * dataset) {

  int nVars = dataset->nVars;
  double * w = model->w;
  double * Xt = dataset->Xt;
  double * y = dataset->y;
  double * d = trainer->d;
  double * g = trainer->g;
  double * Li = dataset->Li;
  double * Lmax = dataset->Lmax;
  double lambda = trainer->lambda;
  int nSamples = dataset->nSamples;
  double precision = trainer->precision;
  int increasing = dataset->increasing;

  double * randVals = dataset->randVals;
  int maxIter = trainer->maxIter;
  int k = trainer->iter;
  int * covered = dataset->covered;
  double nCovered = dataset->nCovered;

  int nextpow2 = dataset->nextpow2;
  int nLevels = dataset->nLevels;
  double * nDescendants = dataset->nDescendants;
  double * unCoveredMatrix = dataset->unCoveredMatrix;
  double * LiMatrix = dataset->LiMatrix;
  double Lmean = dataset->Lmean;

  /* Select next training example */
  double offset = 0;
  int i = 0;
  double u = randVals[k + maxIter];
  double z, Z;
  if(randVals[k] < (double)(nSamples - nCovered)/(double)nSamples) {
    /* Sample fron uncovered guys */
    Z = unCoveredMatrix[nextpow2 * (nLevels - 1)];
    for(int level=nLevels - 1;level >= 0; level--) {
      z = offset + unCoveredMatrix[2 * i + nextpow2 * level];
      if(u < z/Z) {
        i = 2 * i;
      } else {
        offset = z;
        i = 2 * i + 1;
      }
    }
  } else {
    /* Sample from covered guys according to estimate of Lipschitz constant */
    Z = LiMatrix[nextpow2 * (nLevels - 1)] +
        (Lmean + 2 * lambda) *
        (nDescendants[nextpow2 * (nLevels - 1)] -
         unCoveredMatrix[nextpow2 * (nLevels - 1)]);
    for(int level = nLevels - 1; level  >= 0; level--) {
      z = offset + LiMatrix[2 * i + nextpow2 * level] +
          (Lmean + 2 * lambda) *
          (nDescendants[2 * i + nextpow2 * level] -
           unCoveredMatrix[2 * i + nextpow2 * level]);
      if(u < z/Z) {
        i = 2 * i;
      } else {
        offset = z;
        i = 2 * i + 1;
      }
    }
    if(DEBUG) Rprintf("i = %d", i);
  }

  /* Compute current values of needed parameters */

  if (sparse) {
    // TODO(Ishmael): SAG_LipschitzLS_logistic_BLAS.c line 192
  }

  /* Compute derivative of loss */
  double innerProd = 0;
  if (sparse) {
    // TODO(Ishmael): SAG_LipschitzLS_logistic_BLAS.c line 206
  } else {
    innerProd = F77_CALL(ddot)(&nVars, w, &one, &Xt[nVars * i], &one);
  }

  double grad = model->grad(y[i], innerProd);

  /* Update direction */
  double scaling;
  if (sparse) {
    // TODO(Ishmael):  SAG_LipschitzLS_logistic_BLAS.c line 216
  } else {
    scaling = grad - g[i];
    F77_CALL(daxpy)(&nVars, &scaling, &Xt[i * nVars], &one, d, &one);
  }

  /* Store derivative of loss */
  g[i] = grad;

  /* Line-search for Li */
  double Li_old = Li[i];
  if(increasing && covered[i]) Li[i] /= 2;
  double fi = model->loss(y[i], innerProd);

  /* Compute f_new as the function value obtained by taking
   * a step size of 1/Li in the gradient direction */
  double wtx = innerProd;
  double xtx = F77_CALL(ddot)(&nVars, &Xt[i * nVars], &one, &Xt[i * nVars], &one);
  double gg = grad * grad * xtx;
  innerProd = wtx - xtx * grad/Li[i];

  double fi_new = model->loss(y[i], innerProd);
  if(DEBUG) Rprintf("fi = %e, fi_new = %e, gg = %e", fi, fi_new, gg);
  while (gg > precision && fi_new > fi - gg/(2*(Li[i]))) {
    if (DEBUG) {  Rprintf("Lipschitz Backtracking (k = %d, fi = %e, * fi_new = %e, 1/Li = %e)", k +1 ,
                          fi, fi_new, 1/(Li[i]));
    }
    Li[i] *= 2;
    innerProd = wtx - xtx * grad/Li[i];
    fi_new = model->loss(y[i], innerProd);
  }
  if(Li[i] > *Lmax) *Lmax = Li[i];

  /* Update the number of examples that we have seen */
  int ind;
  if (covered[i] == 0) {
    covered[i] = 1;
    nCovered++;
    Lmean = Lmean *((double)(nCovered - 1)/(double)nCovered) +
            Li[i]/(double)nCovered;

    /* Update unCoveredMatrix so we don't sample this guy when looking for a new guy */
    ind = i;
    for(int level = 0; level< nLevels; level++) {
      unCoveredMatrix[ind + nextpow2 * level] -= 1;
      ind = ind/2;
    }
    /* Update LiMatrix so we sample this guy proportional to its Lipschitz constant*/
    ind = i;
    for(int level = 0; level < nLevels; level++) {
      LiMatrix[ind + nextpow2 * level] += Li[i];
      ind = ind/2;
    }
  } else if (Li[i] != Li_old) {
    Lmean = Lmean + (Li[i] - Li_old)/(double)nCovered;

    /* Update LiMatrix with the new estimate of the Lipscitz constant */
    ind = i;
    for(int level = 0; level < nLevels; level++) {
      LiMatrix[ind + nextpow2 * level] += (Li[i] - Li_old);
      ind = ind/2;
    }
  }
  if (DEBUG) {
    for(int ind = 0; ind < nextpow2; ind++) {
      for(int j = 0;j < nLevels; j++) {
        Rprintf("%f ", LiMatrix[ind + nextpow2 * j]);
      }
      //Rprintf("\n");
    }
  }
  /* Compute step size */
  double alpha = ((double)(nSamples - nCovered)/(double)nSamples)/(*Lmax + lambda) +
                 ((double)nCovered/(double)nSamples) * (1/(2*(*Lmax + lambda)) +
                                                        1/(2*(Lmean + lambda)));
  /* Update parameters */
  if (sparse) {
    // TODO(Ishmael): SAG_LipschitzLS_logistic_BLAS.c line 294
  } else {
    scaling = 1 - alpha * lambda;
    F77_CALL(dscal)(&nVars, &scaling, w, &one);
    scaling = -alpha/nCovered;
    F77_CALL(daxpy)(&nVars, &scaling, d, &one, w, &one);
  }

  /* Decrease value of max Lipschitz constant */
  if (increasing) {
    *Lmax *= pow(2.0, -1.0/nSamples);
  }

}
