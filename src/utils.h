#ifndef UTILS_H_
#define UTILS_H_
#include "sag_common.h"

/* Macros */
/* DEBUG MACROS*/
#ifndef R_TRACE
#define R_TRACE( x, ... ) Rprintf(" TRACE @ %s:%d \t" x "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#endif
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


/* Check if R_NilValue */
#ifndef IS_R_NULL
#define IS_R_NULL( x ) (( ( x ) == R_NilValue)? 1: 0)
#endif

/* Prototypes */
double _log_sum_exp(const double * restrict array, const int ar_size);
double log2(double x);
double get_cost_agrad_norm(const double* restrict w, const double* restrict d, const double lambda,
                           const double nCovered, const int nSamples, const int nVars);
#endif /* UTILS_H_ */
