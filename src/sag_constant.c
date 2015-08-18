#include <stdio.h>
#include <errno.h>
#include <string.h>
#include "sag_constant.h"
#include "Matrix.h"
#include "cholmod.h"

#define R_TRACE( x, ... ) Rprintf(" TRACE @ %s:%d \t" x "\n", __FILE__, __LINE__, ##__VA_ARGS__)

/* Constant */
const static int one = 1;
const static int DEBUG = 0;
static inline void _sag_constant_iteration(GlmTrainer * trainer,
                                           GlmModel * model,
                                           Dataset * dataset);
/*============\
| entry-point |
\============*/

/**
 * Logistic regression stochastic average gradient trainer
 *
 * @param w(p, 1) weights
 * @param Xt(p, n) real fature matrix
 * @param y(n, 1) {-1, 1} target matrix
 * @param lambda scalar regularization parameter
 * @param stepSize scalar constant step size
 * @param iVals(max_iter, 1) sequence of examples to choose
 * @param d(p, 1) approximation of average gradient
 * @param g(n, 1) previous derivatives of loss
 * @param covered(n, 1) whether the example has been visited
 * @return optimal weights (p, 1)
 *
 */
SEXP C_sag_constant(SEXP w, SEXP Xt, SEXP y, SEXP lambda,
                    SEXP stepSize, SEXP iVals, SEXP d, SEXP g,
                    SEXP covered, SEXP family, SEXP tol, SEXP sparse) {
  // Initializing garbage collection protection counter
  int nprot = 0;
  /*======\
  | Input |
  \======*/
  /* Initializing dataset */
  Dataset train_set = {.y = REAL(y),
                       .iVals = INTEGER(iVals),
                       .covered = INTEGER(covered),
                       .nCovered = 0,
                       .sparse = *INTEGER(sparse)};
  CHM_SP cXt;
  if (train_set.sparse) {
    cXt = AS_CHM_SP(Xt);
    /* Sparse Array pointers*/
    train_set.ir = cXt->i;
    train_set.jc = cXt->p;
    train_set.Xt = cXt->x;
    /* Sparse Variables */
    train_set.nVars = cXt->nrow;
    train_set.nSamples = cXt->ncol;
    /* Allocate Memory Needed for lazy update */
    train_set.lastVisited = Calloc(train_set.nVars, int);
    train_set.cumSum = Calloc(INTEGER(GET_DIM(iVals))[0], double);
  } else {
    train_set.Xt = REAL(Xt);
    train_set.nSamples = INTEGER(GET_DIM(Xt))[1];
    train_set.nVars = INTEGER(GET_DIM(Xt))[0];
  }
  /* Initializing Trainer */
  GlmTrainer trainer = {.lambda = *REAL(lambda),
                        .alpha = *REAL(stepSize),
                        .c = 1.0,
                        .d = REAL(d),
                        .g = REAL(g),
                        .iter = 0,
                        .maxIter = INTEGER(GET_DIM(iVals))[0],
                        .tol = *REAL(tol)};

  /* Initializing Model */
  // TODO(Ishmael): Model Dispatch should go here

  GlmModel model = {.w = REAL(w)};
  if (DEBUG) Rprintf("data structures initalized.\n");
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
if (DEBUG) Rprintf("Model functions assigned. \n");
  /*===============\
  | Error Checking |
  \===============*/
  if (train_set.nVars != INTEGER(GET_DIM(w))[0]) {
    error("w and Xt must have the same number of rows");
  }
  if (DEBUG) Rprintf(" w and Xt check passed.\n");
  if (train_set.nSamples != INTEGER(GET_DIM(y))[0]) {
    error("number of columns of Xt must be the same as the number of rows in y");
  }
  if (DEBUG) Rprintf(" nSamples and Xt dim check passed.\n");
  if (train_set.nVars != INTEGER(GET_DIM(d))[0]) {
    error("w and d must have the same number of rows");
  }
  if (DEBUG) Rprintf(" w and d dim check passed.\n");
  if (train_set.nSamples != INTEGER(GET_DIM(g))[0]) {
    error("w and g must have the same number of rows");
  }
  if (DEBUG) Rprintf(" w and g dim check passed.\n");
  if (train_set.nSamples != INTEGER(GET_DIM(covered))[0]) {
    error("covered and y must have the same number of rows");
  }
  if (DEBUG) Rprintf(" covered and y dim check passed.\n");
  // TODO(Ishmael): SAG_logistic_BLAS line 62
  if (train_set.sparse && trainer.alpha * trainer.lambda == 1) {
    error("sorry, I don't like it when Xt is sparse and alpha*lambda=1\n");
  }
  if (DEBUG) Rprintf(" sparse and alpha * lambda check passd. \n");
  if (DEBUG) Rprintf("Initial error Checks all passed\n");
  /*==============================\
  | Stochastic Average Gradient   |
  \==============================*/

  /* Counting covered examples*/
  for (int i = 0; i < train_set.nSamples; i++) {
    if (train_set.covered[i] != 0) train_set.nCovered++;
  }

  double cost_grad_norm = get_cost_grad_norm(&trainer, &model, &train_set);
  /* Rprintf("initial cost grad norm %4.6f\n", cost_grad_norm); */
  int stop_condition = 0;
  while (!stop_condition) {
    _sag_constant_iteration(&trainer, &model, &train_set);
    if (trainer.iter % 10000 == 0) {
      Rprintf("Trainer.iter = %d \n", trainer.iter);
    }
    trainer.iter++;
    cost_grad_norm = get_cost_grad_norm(&trainer, &model, &train_set);
    /* if (trainer.iter % 1000 == 0) { */
    /*   Rprintf("Norm of approximate gradient at iteration %d/%d: \t %f \n", trainer.iter, trainer.maxIter, cost_grad_norm); */
    /* } */
    stop_condition = (trainer.iter >= trainer.maxIter) || (cost_grad_norm <= trainer.tol);
    if (stop_condition) {
      Rprintf("Stop condition is satisfied @ iter: %d \n", trainer.iter);
    }
  }
  int convergence_code = 0;  // 0 -- converged.  1 -- did not converge.
  if (cost_grad_norm > trainer.tol) {
    warning("(constant) Optmisation stopped before convergence: %d/%d\n", trainer.iter, trainer.maxIter);
    convergence_code = 1;
  }

  if (train_set.sparse) {
    for(int j = 0; j < train_set.nVars; j++) {
      if (train_set.lastVisited[j]==0) {
        model.w[j] -= trainer.d[j]*train_set.cumSum[trainer.maxIter-1];
      } else {
        model.w[j] -= trainer.d[j]*(train_set.cumSum[trainer.maxIter-1]-train_set.cumSum[train_set.lastVisited[j]-1]);
      }
    }
    double scaling = trainer.c;
    F77_CALL(dscal)(&train_set.nVars,&scaling,model.w,&one);
    Free(train_set.lastVisited);
    Free(train_set.cumSum);
  }

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
  SEXP convergence_code_return = PROTECT(allocVector(INTSXP, 1)); nprot++;
  *INTEGER(convergence_code_return) = convergence_code;
  SEXP iter_return = PROTECT(allocVector(INTSXP, 1)); nprot++;
  *INTEGER(iter_return) = trainer.iter;

  /* Assigning variables to SEXP list */
  SEXP results = PROTECT(allocVector(VECSXP, 6)); nprot++;
  INC_APPLY(SEXP, SET_VECTOR_ELT, results, w_return, d_return, g_return, covered_return, convergence_code_return, iter_return); // in utils.h
  /* Creating SEXP for list names */
  SEXP results_names = PROTECT(allocVector(STRSXP, 6)); nprot++;
  INC_APPLY_SUB(char *, SET_STRING_ELT, mkChar, results_names, "w", "d", "g", "covered", "convergence_code", "iter");
  setAttrib(results, R_NamesSymbol, results_names);

  UNPROTECT(nprot);
  return results;
}

static inline void _sag_constant_iteration(GlmTrainer * trainer,
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
  int i = dataset->iVals[trainer->iter] - 1;
  /* Compute current values of needed parameters */
  if (dataset->sparse && trainer->iter > 0) {
    for (int j = dataset->jc[i]; j < dataset->jc[i+1]; j++) {
      if (dataset->lastVisited[dataset->ir[j]] == 0) {
        model->w[dataset->ir[j]] -= d[dataset->ir[j]] *
          dataset->cumSum[trainer->iter - 1];
      } else {
        model->w[dataset->ir[j]] -= d[dataset->ir[j]] *
          (dataset->cumSum[trainer->iter-1] - dataset->cumSum[dataset->lastVisited[dataset->ir[j]] - 1]);
      }
      dataset->lastVisited[dataset->ir[j]] = trainer->iter;
    }
  }

  /* Compute derivative of loss */
  double innerProd = 0;
  if (dataset->sparse) {
    innerProd = 0;
    for (int j=dataset->jc[i]; j < dataset->jc[i+1]; j++) {
      innerProd += w[dataset->ir[j]] * Xt[j];
    }
    innerProd *= trainer->c;
  } else {
    innerProd = F77_CALL(ddot)(&nVars, w, &one, &Xt[nVars * i], &one);
  }

  double grad = model->grad(y[i], innerProd);

  /* Update direction */
  double scaling = 0;
  if (dataset->sparse) {
    for(int j=dataset->jc[i]; j < dataset->jc[i+1]; j++) {
      d[dataset->ir[j]] += Xt[j]*(grad - g[i]);
    }
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
    trainer->c *= 1 - trainer->alpha * trainer->lambda;
    if (trainer->iter == 0) {
      dataset->cumSum[0] = trainer->alpha/(trainer->c * dataset->nCovered);
    } else {
      dataset->cumSum[trainer->iter] = dataset->cumSum[trainer->iter-1] +
        trainer->alpha/(trainer->c * dataset->nCovered);
    }
  } else {
    scaling = 1 - trainer->alpha * trainer->lambda;
    F77_CALL(dscal)(&nVars, &scaling, w, &one);
    scaling = -trainer->alpha/dataset->nCovered;
    F77_CALL(daxpy)(&nVars, &scaling, d, &one, w, &one);
  }
}

