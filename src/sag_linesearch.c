#include <stdio.h>
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>

const int DEBUG = 0;

/**
 *     Logistic regression stochastic average gradient trainer
 *    
 *     @param w_s(p, 1) weights
 *     @param Xt_s(p, n) real fature matrix
 *     @param y_s(n, 1) {-1, 1} target matrix
 *     @param lambda_s scalar regularization parameters
 *     @param stepSize_s scalar constant step size
 *     @param iVals_s(max_iter, 1) sequence of examples to choose
 *     @param d_s(p, 1) initial approximation of average gradient
 *     @param g_s(n, 1) previous derivatives of loss
 *     @param covered_s(n, 1) whether the example has been visited
 *     @param stepSizeType_s scalar default is 1 to use 1/L, set to 2 to
 *     use 2/(L + n*myu)
 *     @param xtx_s squared norm of features   
 *     @return optimal weights (p, 1)
 */
SEXP SAG_logistic(SEXP w_s, SEXP Xt_s, SEXP y_s, SEXP lambda_s,
                  SEXP stepSize_s, SEXP iVals_s, SEXP d_s, SEXP g_s,
                  SEXP covered_s, SEXP stepSizeType_s, SEXP xtx_s) {

  
  
  return NULL;
}
