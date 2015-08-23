#include "utils.h"

/**
 * Log-Sum-Exp
 *
 * @param array
 * @param ar_size
 *
 * @return logsumexp
 */
double _log_sum_exp(const double * restrict array, const int ar_size) {

  double sum = 0.0;
  double array_max = array[0];

  /* Getting array maximum */
  for (int i = 1; i < ar_size; i++) {
    if (array[i] > array_max) {
      array_max = array[i];
    }
  }
  /* computing exponentials sum */
  for (int i = 0; i < ar_size; i++) {
    sum += exp(array[i] - array_max);
  }
  return array_max + log(sum);
}
// TODO(Ishmael): Consider using R math functions
/**
 * Log base 2
 *
 * @param x real
 *
 * @return
 */ 
double log2(double x) {
  return log(x)/log(2);
}
/* computes cost function's approximate gradient norm */
double get_cost_agrad_norm(double* w, double* d, double lambda,
                          double nCovered, int nSamples, int nVars) {
  /* Normalize approximete gradient by the min of seen example and sample Size.
     This is done to avoid having an artificially small normalized approximate gradient
     when it is initialized at zero */
  double norm_const;
  if (nCovered < 1.0f) {
    norm_const = nSamples;
  } else {
    norm_const = fmin(nCovered, (double)nSamples);
  }

  double cost_grad_norm = 0;
  for(int i = 0; i < nVars; i++) {
    cost_grad_norm += pow(d[i] + norm_const * lambda * w[i], 2.0);
  }
  return sqrt(cost_grad_norm)/norm_const;
}



