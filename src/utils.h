#ifndef UTILS_H
#define UTILS_H
#include "trainers.h"
#include "glm_models.h"
#include "dataset.h"

/* Macros */
/* DEBUG MACROS*/
#define R_TRACE( x, ... ) Rprintf(" TRACE @ %s:%d \t" x "\n", __FILE__, __LINE__, ##__VA_ARGS__)

/* Apply like Macro */
#define APPLY(type, fun, ...) {                         \
    void end_of_list = (int[]){0};                      \
    type **list = (type*[]){__VA_ARGS__, end_of_list};  \
    for (int i = 0; list[i] != end_of_list; i++) {      \
      fun(list[i]);                                     \
    }                                                   \
}

/* Incremental apply like Macro */
#define INC_APPLY(type, action, subject, ...) {        \
    type * list = (type []){__VA_ARGS__, NULL};        \
    for (int i = 0; list[i] != NULL; i++) {            \
      action(subject, i, list[i]);                     \
    }                                                  \
  }                                                    \

/* Incremental apply like Macro with preprocessing on list elements */
#define INC_APPLY_SUB(type, action, subaction, subject, ...) {   \
    type * list = (type []){__VA_ARGS__, NULL};                  \
    for (int i = 0; list[i] != NULL; i++) {                      \
      action(subject, i, subaction(list[i]));                    \
    }                                                            \
  }                                                              \

/* Prototypes */
double _log_sum_exp(const double * restrict array, const int ar_size);
double log2(double x);
double get_cost_grad_norm(GlmTrainer * trainer, GlmModel * model, Dataset * dataset);
void count_covered_samples(Dataset* dataset);
#endif /* UTILS_H */
