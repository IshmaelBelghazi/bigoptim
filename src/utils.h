#ifndef UTILS_H
#define UTILS_H

/* Macros */
/* Apply like Macro */
#define APPLY(type, fun, ...) {                         \
    void end_of_list = (int[]){0};                      \
    type **list = (type*[]){__VA_ARGS__, end_of_list};  \
    for (int i = 0; list[i] != end_of_list; i++) {      \
      fun(list[i]);                                     \
    }                                                   \
}

/* Assigning Several SEXP to a vector SEXP */
#define ASSIGN_TO_R_VECTOR(vector, ...) {         \
    SEXP * list = (SEXP []){__VA_ARGS__, NULL};   \
    for (int i = 0; list[i] != NULL; i++) {       \
      SET_VECTOR_ELT(vector, i, list[i]);         \
    }                                             \
  }                                               \

/* */


//#define ASSIGN_TO_R_VECTOR()

/* Prototypes */
double _log_sum_exp(const double * restrict array, const int ar_size);
double log2(double x);

#endif /* UTILS_H */
