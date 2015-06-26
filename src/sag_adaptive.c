#include <stdio.h>
#include <math.h> // TODO(Ishmael): Consider using R math functions
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "utils.h"

// TODO(Ishmael): Consider using R math functions
const static int DEBUG = 0;
const static int one = 1;
const static int sparse = 0;
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
SEXP C_sag_adaptive(SEXP w_s, SEXP Xt_s, SEXP y_s, SEXP lambda_s, SEXP Lmax_s,
                    SEXP Li_s, SEXP randVals_s, SEXP d_s, SEXP g_s, SEXP covered_s,
                    SEXP increasing_s) {
  // initializing protection counter
  int nprot = 0;
  /* Variables */
  // TODO(Ishmael): This is messy. Clean it. 
  // int temp;
  // int  * lastVisited;
  // int i, j;
  // size_t * jc,* ir;
    
  double alpha;
  // double c=1;
  // double * cumSum;
  
  /*======\
  | Input |
  \======*/
  
  double * w = REAL(w_s);
  double * Xt = REAL(Xt_s);
  double * y = REAL(y_s);
  double lambda = *REAL(lambda_s);
  double * Lmax = REAL(Lmax_s);
  double * Li = REAL(Li_s);
  double * randVals = REAL(randVals_s);
  double * d = REAL(d_s);
  double * g = REAL(g_s);
  int * covered = INTEGER(covered_s);
  int increasing = *INTEGER(increasing_s);

  /* Compute Sizes */
  int nSamples = INTEGER(GET_DIM(Xt_s))[1];
  int nVars = INTEGER(GET_DIM(Xt_s))[0];
  int maxIter = INTEGER(GET_DIM(randVals_s))[0];
  if (DEBUG) Rprintf("nSamples: %d\n", nSamples);
  if (DEBUG) Rprintf("nVars: %d\n", nVars);
  if (DEBUG) Rprintf("maxIter: %d\n", maxIter);

  double precision = 1.490116119384765625e-8;
  double * xtx = Calloc(nSamples, double);


  /* Error Checking */
  if (nVars != INTEGER(GET_DIM(w_s))[0]) {
    error("w and Xt must have the same number of rows");
  }
  if (nSamples != INTEGER(GET_DIM(y_s))[0]) {
    error("number of columns of Xt must be the same as the number of rows in y");
  }
  if (nVars != INTEGER(GET_DIM(d_s))[0]) {
    error("w and d must have the same number of rows");
  }
  if (nSamples != INTEGER(GET_DIM(g_s))[0]) {
    error("w and g must have the same number of rows");
  }
  if (nSamples != INTEGER(GET_DIM(covered_s))[0]) {
    error("covered and y must hvae the same number of rows");
  }
  // TODO(Ishmael): SAG_LipschitzLS_logistic_BLAS line 78
  if (sparse && alpha * lambda == 1) {
    error("Sorry, I don't like it when Xt is sparse and alpha*lambda=1\n");
  }
  /*============================\
  | Stochastic Average Gradient |
  \============================*/
  /* Allocate memory needed for lazy updates*/
  if (sparse) {
    // TODO(Ishmael): SAG_LipschitzLS_logistic_BLAS line 89
  }
   
  /* Compute mean of covered variables */
  double Lmean = 0;
  double nCovered = 0;
  for(int i = 0;i < nSamples; i++) {
    if (covered[i] != 0) {
      nCovered++;
      Lmean += Li[i];
    }
  }
  
  if(nCovered > 0) {
    Lmean /= nCovered;
  }
  
  for (int i = 0; i < nSamples; i++) {
    if (sparse) {
      // TODO(Ishmael): SAGlineSearch_logistic_BLAS.c line 103
    } else {
      // TODO(Ishmael): use a higher level BLAS OPERATION
      xtx[i] = F77_CALL(ddot)(&nVars, &Xt[i * nVars], &one, &Xt[i * nVars], &one);
    }
  }
  /* Do the O(n log n) initialization of the data structures
     will allow sampling in O(log(n)) time */
  int nextpow2 = pow(2, ceil(log2(nSamples)/log2(2)));
  int nLevels = 1 + (int)ceil(log2(nSamples));
  if (DEBUG) Rprintf("next power of 2 is: %d\n",nextpow2);
  if (DEBUG) Rprintf("nLevels = %d\n",nLevels);
  /* Counts number of descendents in tree */
  double * nDescendants = Calloc(nextpow2 * nLevels,double); 
  /* Counts number of descenents that are still uncovered */
  double * unCoveredMatrix = Calloc(nextpow2 * nLevels, double); 
  /* Sums Lipschitz constant of loss over descendants */
  double * LiMatrix = Calloc(nextpow2 * nLevels, double); 
  for(int i = 0; i < nSamples; i++) {
    nDescendants[i] = 1;
    if (covered[i]) {
        LiMatrix[i] = Li[i];
    } else {
      unCoveredMatrix[i] = 1;
    }
    
  }
  int levelMax = nextpow2;
  for (int level = 1; level < nLevels; level++) {
    levelMax = levelMax/2;
    for(int i = 0; i < levelMax; i++) {
      nDescendants[i + nextpow2 * level] = nDescendants[ 2 * i + nextpow2 * (level - 1)] +
                                           nDescendants[ 2 * i + 1 + nextpow2 * (level - 1)];
      LiMatrix[i + nextpow2 * level] = LiMatrix[2 * i + nextpow2 * (level - 1)] +
                                       LiMatrix[ 2 * i + 1 + nextpow2 * (level - 1)];
      unCoveredMatrix[i + nextpow2 * level] = unCoveredMatrix[2 * i + nextpow2 * (level - 1)] +
                                              unCoveredMatrix[2 * i + 1 + nextpow2 * (level - 1)];
    }
  }

  for(int k = 0; k < maxIter; k++) {
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
      if(DEBUG) Rprintf("i = %d\n", i);
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
    
    double sig = -y[i]/(1 + exp(y[i] * innerProd));
        
    /* Update direction */
    double scaling;
    if (sparse) {
      // TODO(Ishmael):  SAG_LipschitzLS_logistic_BLAS.c line 216
    } else {
      scaling = sig-g[i];
      F77_CALL(daxpy)(&nVars, &scaling, &Xt[i * nVars], &one, d, &one);
    }
    
    /* Store derivative of loss */
    g[i] = sig;

    /* Line-search for Li */
    double Li_old = Li[i];
    if(increasing && covered[i]) Li[i] /= 2;
    double fi = log(1 + exp(-y[i] * innerProd));

    /* Compute f_new as the function value obtained by taking 
     * a step size of 1/Li in the gradient direction */
    double wtx = innerProd;
    double gg = sig * sig * xtx[i];
    innerProd = wtx - xtx[i] * sig/Li[i];
    double fi_new = log(1 + exp(-y[i] * innerProd));
    if(DEBUG) Rprintf("fi = %e, fi_new = %e, gg = %e\n", fi, fi_new, gg);
    while (gg > precision && fi_new > fi - gg/(2*(Li[i]))) {
      if (DEBUG) {  Rprintf("Lipschitz Backtracking (k = %d, fi = %e, * fi_new = %e, 1/Li = %e)\n", k +1 ,
                            fi, fi_new, 1/(Li[i]));
      }
      Li[i] *= 2;
      innerProd = wtx - xtx[i] * sig/Li[i];
      fi_new = log(1 + exp(-y[i] * innerProd));            
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
        Rprintf("\n");
      }
    }
     /* Compute step size */
    alpha = ((double)(nSamples - nCovered)/(double)nSamples)/(*Lmax + lambda) +
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
      *Lmax *= pow(2.0,-1.0/nSamples);
    }
  }

  if (sparse) {
    // TODO(Ishmael): SAG_LipschitzLS_logistic_BLAS.c line 315
  }

  /* Freeing allocated variables */
  Free(xtx);
  Free(nDescendants);
  Free(unCoveredMatrix);
  Free(LiMatrix);

  /*=======\
  | Return |
  \=======*/
  
  /* Preparing return variables  */
  SEXP w_return = PROTECT(allocMatrix(REALSXP, nVars, 1)); nprot++;
  Memcpy(REAL(w_return), w, nVars);
  SEXP d_return = PROTECT(allocMatrix(REALSXP, nVars, 1)); nprot++;
  Memcpy(REAL(d_return), d, nVars);
  SEXP g_return = PROTECT(allocMatrix(REALSXP, nSamples, 1)); nprot++;
  Memcpy(REAL(g_return), g, nSamples);
  SEXP covered_return = PROTECT(allocMatrix(INTSXP, nSamples, 1)); nprot++;
  Memcpy(INTEGER(covered_return), covered, nSamples);

  /* Assigning variables to list */
  SEXP results = PROTECT(allocVector(VECSXP, 4)); nprot++;
  SET_VECTOR_ELT(results, 0, w_return);
  SET_VECTOR_ELT(results, 1, d_return);
  SET_VECTOR_ELT(results, 2, g_return);
  SET_VECTOR_ELT(results, 3, covered_return);
  /* Setting list names */
  SEXP results_names = PROTECT(allocVector(STRSXP, 4)); nprot++;
  const char * names[4] = {"w", "d", "g", "covered"};
  for (int i = 0; i < 4; i++) {
  SET_STRING_ELT(results_names, i, mkChar(names[i]));
  }
  setAttrib(results, R_NamesSymbol, results_names);
  // SEXP results = PROTECT(allocVector(VECSXP, 3)); nprot++;
  UNPROTECT(nprot);

  return results;
}
