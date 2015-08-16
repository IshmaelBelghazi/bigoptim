#include <stdio.h>
//#include <math.h>
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "utils.h"
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"

// TODO(Ishmael): Consider using R math functions
const static int DEBUG = 0;
const static int sparse = 0;
const static int one = 1;
const static double precision = 1.490116119384765625e-8;

static inline void _sag_adaptive_iteration(GlmTrainer * trainer,
                                           GlmModel * model,
                                           Dataset * dataset);
/**
 *   Stochastic Average Gradient Descent with line-search and adaptive
 *   lipschitz sampling
 *
 *   @param w_s (p, 1) real weights
 *   @param Xt_s (p, n) real features Matrix
 *   @param y_s (m, 1) {-1, 1} targets Matrix
 *   @param lambda_s scalar regularization parameter
 *   @param Lmax_s scalar initial approximation of global Lipschitz constants
 *   @param Li_s (n, 1) initial approximation of inidividual lipschitz constants
 *   @param randVals (maxiter, 2) - sequence of random values for the
 *   algorithm to use
 *   @param d_s (p, 1) initial approximation of average gradient
 *   @param g_s (n, 1) previousd derivatives of loss
 *
 *   @param covered_s  d(p,1) initial approximation of average gradient (should be sum of previous gradients)
 *   @param increasing_s  scalar default is 1 to allow the Lipscthiz constants to increase, set to 0 to only allow them to decrease
 *
 *   @return optimal weights (p, 1)
 */
SEXP C_sag_adaptive(SEXP w, SEXP Xt, SEXP y, SEXP lambda, SEXP Lmax,
                    SEXP Li, SEXP randVals, SEXP d, SEXP g, SEXP covered,
                    SEXP increasing, SEXP family, SEXP tol) {
  // initializing protection counter
  int nprot = 0;
  /* Variables */
  // TODO(Ishmael): This is messy. Clean it.
  // int temp;
  // int  * lastVisited;
  // int i, j;
  // size_t * jc,* ir;

  // double c=1;
  // double * cumSum;

  /*======\
  | Input |
  \======*/

  /* Initializing dataset */
  Dataset train_set = {.Xt = REAL(Xt),
                       .y = REAL(y),
                       .randVals = REAL(randVals),
                       .nSamples = INTEGER(GET_DIM(Xt))[1],
                       .nVars = INTEGER(GET_DIM(Xt))[0],
                       .Lmax = REAL(Lmax),
                       .Li = REAL(Li),
                       .covered = INTEGER(covered),
                       .increasing = *INTEGER(increasing),
                       .sparse = sparse};

  /* Initialzing trainer */
  GlmTrainer trainer = {.lambda = *REAL(lambda),
                        .d = REAL(d),
                        .g = REAL(g),
                        .iter = 0,
                        .maxIter = INTEGER(GET_DIM(randVals))[0],
                        .precision = precision,
                        .tol = *REAL(tol)};

   /* Initializing Model */
   GlmModel model = {.w = REAL(w)};
   /* Choosing family */
   switch (*INTEGER(family)) {
   case GAUSSIAN:
     model.loss = gaussian_loss;
     model.grad = gaussian_loss_grad;
     break;
   case BINOMIAL:
     model.loss = binomial_loss;
     model.grad = binomial_loss_grad;
     break;
   case EXPONENTIAL:
     model.loss = exponential_loss;
     model.grad = exponential_grad;
     break;
   case POISSON:
     model.loss = poisson_loss;
     model.grad = poisson_loss_grad;
     break;
   default:
     error("Unrecognized glm family");
   }

  /* Error Checking */
  if (train_set.nVars != INTEGER(GET_DIM(w))[0]) {
    error("w and Xt must have the same number of rows");
  }
  if (train_set.nSamples != INTEGER(GET_DIM(y))[0]) {
    error("number of columns of Xt must be the same as the number of rows in y");
  }
  if (train_set.nVars != INTEGER(GET_DIM(d))[0]) {
    error("w and d must have the same number of rows");
  }
  if (train_set.nSamples != INTEGER(GET_DIM(g))[0]) {
    error("w and g must have the same number of rows");
  }
  if (train_set.nSamples != INTEGER(GET_DIM(covered))[0]) {
    error("covered and y must hvae the same number of rows");
  }
  // TODO(Ishmael): SAG_LipschitzLS_logistic_BLAS line 78
  /* if (sparse && alpha * lambda == 1) { */
  /*   error("Sorry, I don't like it when Xt is sparse and alpha*lambda=1\n"); */
  /* } */
  /*============================\
  | Stochastic Average Gradient |
  \============================*/
  /* Allocate memory needed for lazy updates*/
  if (train_set.sparse) {
    // TODO(Ishmael): SAG_LipschitzLS_logistic_BLAS line 89
  }

  /* Compute mean of covered variables */
  double Lmean = 0;
  double nCovered = 0;
  for(int i = 0; i < train_set.nSamples; i++) {
    if (train_set.covered[i] != 0) {
      nCovered++;
      Lmean += train_set.Li[i];
    }
  }

  if(nCovered > 0) {
    Lmean /= nCovered;
  }

  /* Do the O(n log n) initialization of the data structures
     will allow sampling in O(log(n)) time */
  int nextpow2 = pow(2, ceil(log2(train_set.nSamples)/log2(2)));
  int nLevels = 1 + (int)ceil(log2(train_set.nSamples));
  if (DEBUG) Rprintf("next power of 2 is: %d\n",nextpow2);
  if (DEBUG) Rprintf("nLevels = %d\n",nLevels);
  /* Counts number of descendents in tree */
  double * nDescendants = Calloc(nextpow2 * nLevels, double);
  /* Counts number of descenents that are still uncovered */
  double * unCoveredMatrix = Calloc(nextpow2 * nLevels, double);
  /* Sums Lipschitz constant of loss over descendants */
  double * LiMatrix = Calloc(nextpow2 * nLevels, double);
  for (int i = 0; i < train_set.nSamples; i++) {
    nDescendants[i] = 1;
    if (train_set.covered[i]) {
        LiMatrix[i] = train_set.Li[i];
    } else {
      unCoveredMatrix[i] = 1;
    }
  }

  int levelMax = nextpow2;
  for (int level = 1; level < nLevels; level++) {
    levelMax = levelMax/2;
    for (int i = 0; i < levelMax; i++) {
      nDescendants[i + nextpow2 * level] = nDescendants[ 2 * i + nextpow2 * (level - 1)] +
                                           nDescendants[ 2 * i + 1 + nextpow2 * (level - 1)];
      LiMatrix[i + nextpow2 * level] = LiMatrix[2 * i + nextpow2 * (level - 1)] +
                                       LiMatrix[ 2 * i + 1 + nextpow2 * (level - 1)];
      unCoveredMatrix[i + nextpow2 * level] = unCoveredMatrix[2 * i + nextpow2 * (level - 1)] +
                                              unCoveredMatrix[2 * i + 1 + nextpow2 * (level - 1)];
    }
  }

  /* Continuing dataset initialisation */
  train_set.Lmean = Lmean;
  train_set.nCovered = nCovered;
  train_set.nLevels = nLevels;
  train_set.nextpow2 = nextpow2;
  train_set.nDescendants = nDescendants;
  train_set.unCoveredMatrix = unCoveredMatrix;
  train_set.LiMatrix = LiMatrix;

  //double cost_grad_norm = get_cost_grad_norm(&trainer, &model, &train_set);
  double cost_grad_norm = 1.0;
  int stop_condition = 0;
  while (!stop_condition) {
    _sag_adaptive_iteration(&trainer, &model, &train_set);
    trainer.iter++;
    //cost_grad_norm = get_cost_grad_norm(&trainer, &model, &train_set);
    cost_grad_norm = 1.0;
    /* if (trainer.iter % 1000 == 0) { */
    /*   Rprintf("Norm of approximate gradient at iteration %d/%d: \t %f \n", trainer.iter, trainer.maxIter, cost_grad_norm); */
    /* } */
    stop_condition = (trainer.iter >= trainer.maxIter) || (cost_grad_norm <= trainer.tol);
    if (stop_condition) {
      Rprintf("Stop condition is satisfied @ iter: %d \n", trainer.iter);
    }
  }
  int convergence_code = 0;
  if (cost_grad_norm > trainer.tol) {
    warning("(LS) Optmisation stopped before convergence: %d/%d\n", trainer.iter, trainer.maxIter);
    convergence_code = 1;
  }

  if (train_set.sparse) {
    // TODO(Ishmael): SAG_LipschitzLS_logistic_BLAS.c line 315
  }

  /* Freeing allocated variables */
  Free(nDescendants);
  Free(unCoveredMatrix);
  Free(LiMatrix);

  /*=======\
  | Return |
  \=======*/

  /* Preparing return variables  */
  SEXP w_return = PROTECT(allocMatrix(REALSXP, train_set.nVars, 1)); nprot++;
  Memcpy(REAL(w_return), model.w, train_set.nVars);
  SEXP d_return = PROTECT(allocMatrix(REALSXP, train_set.nVars, 1)); nprot++;
  Memcpy(REAL(d_return), trainer.d, train_set.nVars);
  SEXP g_return = PROTECT(allocMatrix(REALSXP, train_set.nSamples, 1)); nprot++;
  Memcpy(REAL(g_return), trainer.g, train_set.nSamples);
  SEXP covered_return = PROTECT(allocMatrix(INTSXP, train_set.nSamples, 1)); nprot++;
  Memcpy(INTEGER(covered_return), train_set.covered, train_set.nSamples);

  /* Assigning variables to SEXP list */
  SEXP results = PROTECT(allocVector(VECSXP, 4)); nprot++;
  INC_APPLY(SEXP, SET_VECTOR_ELT, results, w_return, d_return, g_return, covered_return); // in utils.h
  /* Creating SEXP for list names */
  SEXP results_names = PROTECT(allocVector(STRSXP, 4)); nprot++;
  INC_APPLY_SUB(char *, SET_STRING_ELT, mkChar, results_names, "w", "d", "g", "covered");
  setAttrib(results, R_NamesSymbol, results_names);

  UNPROTECT(nprot);
  return results;
}

static inline void _sag_adaptive_iteration(GlmTrainer * trainer,
                                           GlmModel * model,
                                           Dataset * dataset) {

  double * w = model->w;
  double * Xt = dataset->Xt;
  double * y = dataset->y;
  double * d = trainer->d;
  double * g = trainer->g;
  double * Li = dataset->Li;
  double * Lmax = dataset->Lmax;
  double lambda = trainer->lambda;

  int nSamples = dataset->nSamples;
  int nVars = dataset->nVars;

  double precision = trainer->precision;
  int increasing = dataset->increasing;
  int sparse = dataset->sparse;

  double * randVals = dataset->randVals;
  int maxIter = trainer->maxIter;
  int k = trainer->iter;
  int * covered = dataset->covered;

  int nextpow2 = dataset->nextpow2;
  int nLevels = dataset->nLevels;
  double * nDescendants = dataset->nDescendants;
  double * unCoveredMatrix = dataset->unCoveredMatrix;
  double * LiMatrix = dataset->LiMatrix;

  /* Select next training example */
  double offset = 0;
  int i = 0;
  double u = randVals[k + maxIter];

  double rhs_cond = (double)(nSamples - dataset->nCovered)/(double)nSamples;
  double z, Z;
  if (randVals[k] < rhs_cond) {
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
        (dataset->Lmean + 2 * lambda) *
        (nDescendants[nextpow2 * (nLevels - 1)] -
         unCoveredMatrix[nextpow2 * (nLevels - 1)]);
    for (int level = nLevels - 1; level  >= 0; level--) {
      z = offset + LiMatrix[2 * i + nextpow2 * level] +
          (dataset->Lmean + 2 * lambda) *
          (nDescendants[2 * i + nextpow2 * level] -
           unCoveredMatrix[2 * i + nextpow2 * level]);
      if(u < z/Z) {
        i = 2 * i;
      } else {
        offset = z;
        i = 2 * i + 1;
      }
    }
  }

  /* Compute current values of needed parameters */

  if (sparse) {
    // TODO(Ishmael): SAG_LipschitzLS_logistic_BLAS.c line 192
  }
  /* for (int l = 0; l < nVars; l++) { */
  /*   Rprintf("w[%d]= %4.4f\n", l, w[l]); */
  /*   Rprintf("Xt[%d]= %4.4f\n", l, Xt[nVars * i + l]); */
  /* } */

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
    dataset->nCovered++;
    dataset->Lmean = dataset->Lmean *((double)(dataset->nCovered - 1)/(double)dataset->nCovered) +
            Li[i]/(double)dataset->nCovered;

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
    dataset->Lmean = dataset->Lmean + (Li[i] - Li_old)/(double)dataset->nCovered;

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
  double alpha = ((double)(nSamples - dataset->nCovered)/(double)nSamples)/(*Lmax + lambda) +
                 ((double)dataset->nCovered/(double)nSamples) * (1/(2*(*Lmax + lambda)) +
                                                        1/(2*(dataset->Lmean + lambda)));


  /* Update parameters */
  if (sparse) {
    // TODO(Ishmael): SAG_LipschitzLS_logistic_BLAS.c line 294
  } else {
    scaling = 1 - alpha * lambda;
    F77_CALL(dscal)(&nVars, &scaling, w, &one);
    scaling = -alpha/dataset->nCovered;
    F77_CALL(daxpy)(&nVars, &scaling, d, &one, w, &one);
  }
  /* Decrease value of max Lipschitz constant */
  if (increasing) {
    *Lmax *= pow(2.0, -1.0/nSamples);
  }

}

