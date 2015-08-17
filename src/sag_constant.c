#include "sag_constant.h"
/* Constant */
const static int one = 1;
const static int DEBUG = 0;
const static int sparse = 0;

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
 */
SEXP C_sag_constant(SEXP w, SEXP Xt, SEXP y, SEXP lambda,
                    SEXP stepSize, SEXP iVals, SEXP d, SEXP g,
                    SEXP covered, SEXP family, SEXP tol) {

  // Initializing garbage collection protection counter
  int nprot = 0;
  /*======\
  | Input |
  \======*/

  /* Initializing dataset */
  Dataset train_set = {.Xt = REAL(Xt),
                       .y = REAL(y),
                       .iVals = INTEGER(iVals),
                       .covered = INTEGER(covered),
                       .nCovered = 0,
                       .nSamples = INTEGER(GET_DIM(Xt))[1],
                       .nVars = INTEGER(GET_DIM(Xt))[0],
                       .sparse = sparse};
  /* Initializing Trainer */
  GlmTrainer trainer = {.lambda = *REAL(lambda),
                        .alpha = *REAL(stepSize),
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
  /* Allocate Memory Needed for lazy update */
  if (sparse) {
  // TODO(Ishmael): If (sparse) line 72 in SAG_logistic_BLAS
  }
  /* Counting*/
  for (int i = 0; i < train_set.nSamples; i++) {
    if (train_set.covered[i] != 0) train_set.nCovered++;
  }

  double cost_grad_norm = get_cost_grad_norm(&trainer, &model, &train_set);
  /* Rprintf("initial cost grad norm %4.6f\n", cost_grad_norm); */
  int stop_condition = 0;
  while (!stop_condition) {
    _sag_constant_iteration(&trainer, &model, &train_set);
    //Rprintf("Trainer.iter = %d \n", trainer.iter);
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

