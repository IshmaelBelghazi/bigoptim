#include "entrypoint-sag_constant.h"

/* Constant */
const static int DEBUG = 0;

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
  count_covered_samples(&train_set);
  /* Training */
  _sag_constant(&trainer, &model, &train_set);

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
  *INTEGER(convergence_code_return) = -1;//convergence_code;
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

