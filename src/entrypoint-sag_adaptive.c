#include "entrypoint-sag_adaptive.h"

const static int DEBUG = 0;
const static double precision = 1.490116119384765625e-8;

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
SEXP C_sag_adaptive(SEXP wInit, SEXP Xt, SEXP y, SEXP lambda,
                    SEXP LmaxInit, SEXP LiInit, SEXP randVals, SEXP dInit,
                    SEXP gInit, SEXP coveredInit, SEXP increasing, SEXP family,
                    SEXP tol, SEXP sparse) {
  /* initializing protection counter */
  int nprot = 0;

  /* Duplicating objects to be modified */
  SEXP w = PROTECT(duplicate(wInit)); nprot++;
  SEXP d = PROTECT(duplicate(dInit)); nprot++;
  SEXP g = PROTECT(duplicate(gInit)); nprot++;
  SEXP covered = PROTECT(duplicate(coveredInit)); nprot++;
  SEXP Lmax = PROTECT(duplicate(LmaxInit)); nprot++;
  SEXP Li = PROTECT(duplicate(LiInit)); nprot++;

  /*======\
  | Input |
  \======*/

  /* Initializing dataset */

  Dataset train_set = {.y = REAL(y),
                       .randVals = REAL(randVals),
                       .Lmax = REAL(Lmax),
                       .Li = REAL(Li),
                       .covered = INTEGER(covered),
                       .nCovered = 0,
                       .increasing = *INTEGER(increasing),
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
    train_set.cumSum = Calloc(INTEGER(GET_DIM(randVals))[0], double);
  } else {
    train_set.Xt = REAL(Xt);
    train_set.nSamples = INTEGER(GET_DIM(Xt))[1];
    train_set.nVars = INTEGER(GET_DIM(Xt))[0];
  }
  /* Initializing Trainer */
  GlmTrainer trainer = {.lambda = *REAL(lambda),
                        .d = REAL(d),
                        .g = REAL(g),
                        .iter_count = 0,
                        .maxIter = INTEGER(GET_DIM(randVals))[0],
                        .tol = *REAL(tol),
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
  if (DEBUG) R_TRACE("next power of 2 is: %d\n",nextpow2);
  if (DEBUG) R_TRACE("nLevels = %d\n",nLevels);
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
  /* Training */
  sag_adaptive(&trainer, &model, &train_set);

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


