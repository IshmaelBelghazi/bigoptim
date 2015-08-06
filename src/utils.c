#include <stdio.h>
#include <Rmath.h>
#include "trainers.h"

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
/* compute gradient norm */
double get_cost_grad_norm(GlmTrainer * trainer, GlmModel * model, Dataset * dataset) {
  double cost_grad_norm = 0;
  for(int i = 0; i < dataset->nVars; i++) {
    cost_grad_norm += pow(trainer->d[i] + dataset->nCovered * trainer->lambda * model->w[i], 2.0);
  }
  return sqrt(cost_grad_norm)/dataset->nCovered;


}
