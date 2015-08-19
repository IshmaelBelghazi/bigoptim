#include "entrypoint-sag_linesearch.h"
#include "sag_linesearch.h"

const static double precision = 1.490116119384765625e-8;

/**
 *     Logistic regression stochastic average gradient trainer
 *
 *     @param w(p, 1) weights
 *     @param Xt(p, n) real feature matrix
 *     @param y(n, 1) {-1, 1} target matrix
 *     @param lambda scalar regularization parameters
 *     @param stepSize scalar constant step size
 *     @param iVals(max_iter, 1) sequence of examples to choose
 *     @param d(p, 1) initial approximation of average gradient
 *     @param g(n, 1) previous derivatives of loss
 *     @param covered(n, 1) whether the example has been visited
 *     @param stepSizeType scalar default is 1 to use 1/L, set to 2 to
 *     use 2/(L + n*myu)
 *     @return optimal weights (p, 1)
 */
SEXP C_sag_linesearch(SEXP w, SEXP Xt, SEXP y, SEXP lambda,
                      SEXP stepSize, SEXP iVals, SEXP d, SEXP g,
                      SEXP covered, SEXP stepSizeType, SEXP family, SEXP tol,
                      SEXP sparse) {
  // Initializing protection counter
  int nprot = 0;

  /*======\
  | Input |
  \======*/

  /* Initializing dataset */
  Dataset train_set = {.y = REAL(y),
                       .iVals = INTEGER(iVals),
                       .covered = INTEGER(covered),
                       .nCovered = 0,
                       .sparse = *INTEGER(sparse),
                       .Li = REAL(stepSize)};

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
                        .tol = *REAL(tol),
                        .stepSizeType = *INTEGER(stepSizeType),
                        .precision = precision};

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


  /*===============\
  | Error Checking |
  \===============*/
  if ( train_set.nVars != INTEGER(GET_DIM(w))[0]) {
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
    error("covered and y must have the same number of rows");
  }
  // TODO(Ishmael): SAGlineSearch_logistic_BLAS.c line 72
  /* if (sparse && alpha * lambda == 1) { // BUG(Ishmael): BUG is mark's */
  /*                   // code alpha is not declared yet. */
  /*   error("Sorry, I don't like it when Xt is sparse and alpha*lambda=1\n"); */
  /* } */

  for(int i = 0; i < train_set.nSamples; i++) {
    if (train_set.covered[i]!=0) train_set.nCovered++;
  }


  /*============================\
  | Stochastic Average Gradient |
  \============================*/

  /* Counting*/
  for (int i = 0; i < train_set.nSamples; i++) {
    if (train_set.covered[i] != 0) train_set.nCovered++;
  }
  _sag_linesearch(&trainer, &model, &train_set);
  /*=======\
  | Return |
  \=======*/
  /* Preparing return variables  */
  SEXP w_return = PROTECT(allocMatrix(REALSXP, train_set.nVars, 1)); nprot++;
  Memcpy(REAL(w_return), model.w, train_set.nVars);
  SEXP d_return = PROTECT(allocMatrix(REALSXP, train_set.nVars, 1)); nprot++;
  Memcpy(REAL(d_return), trainer.d, train_set.nVars);
  // TODO(Ishmael): check g dimesionality
  SEXP g_return = PROTECT(allocMatrix(REALSXP, train_set.nSamples, 1)); nprot++;
  Memcpy(REAL(g_return), trainer.g, train_set.nSamples);
  SEXP covered_return = PROTECT(allocMatrix(INTSXP, train_set.nSamples, 1)); nprot++;
  Memcpy(INTEGER(covered_return), train_set.covered, train_set.nSamples);
  SEXP stepSize_return = PROTECT(allocVector(REALSXP, 1)); nprot++;
  *REAL(stepSize_return) = *train_set.Li;
  SEXP convergence_code_return = PROTECT(allocVector(INTSXP, 1)); nprot++;
  *INTEGER(convergence_code_return) = -1;

  /* Assigning variables to SEXP list */
  SEXP results = PROTECT(allocVector(VECSXP, 6)); nprot++;
  INC_APPLY(SEXP, SET_VECTOR_ELT, results, w_return, d_return, g_return, covered_return, stepSize_return, convergence_code_return); // in utils.h
  /* Creating SEXP for list names */
  SEXP results_names = PROTECT(allocVector(STRSXP, 6)); nprot++;
  INC_APPLY_SUB(char *, SET_STRING_ELT, mkChar, results_names, "w", "d", "g", "covered", "stepSize","convergence_code");
  setAttrib(results, R_NamesSymbol, results_names);

  UNPROTECT(nprot);
  return results;
}


