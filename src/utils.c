#include <Rmath.h>

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
