#include "sag_common.h"
/* ENTRY POINTS -- START */
#include "entrypoint-glm.h"
#include "entrypoint-sag.h"
#include "entrypoint-sag_fit.h"
/* ENTRY POINTS -- END */

#include <R_ext/Rdynload.h>

/** utitlity macro in registering native routines */
#define CALLDEF(name, n)  {#name, (DL_FUNC) &name, n}

/** function that registers native routines  */
static R_CallMethodDef CallEntries[] = {

  /* Cost and Gradient functions*/
  CALLDEF(C_binomial_cost, 4),
  CALLDEF(C_binomial_cost_grad, 4),
  CALLDEF(C_gaussian_cost, 4),
  CALLDEF(C_gaussian_cost_grad, 4),
  CALLDEF(C_exponential_cost, 4),
  CALLDEF(C_exponential_cost_grad, 4),
  CALLDEF(C_poisson_cost, 4),
  CALLDEF(C_poisson_cost_grad, 4),
  /* SAG fit*/
  CALLDEF(C_sag_fit, 18),
  /* SAG with warm-starting */
  CALLDEF(C_sag, 17),
  {NULL, NULL, 0}
};

/* local cholmod_common struct*/
cholmod_common c;

/** This is the CHOLMOD error handler from lme4*/
void attribute_hidden bigoptim_R_cholmod_error(int status,
                                               const char *file,
                                               int line,
                                               const char *message) {
  if (status < 0) {
#ifdef Failure_in_matching_Matrix
    M_cholmod_defaults(&c);
    c.final_ll = 1;
#endif

    error("Cholmod error '%s' at file:%s, line %d", message, file, line);
  } else {
    warning("Cholmod warning '%s' at file:%s, line %d",
            message, file, line);
  }
}

#ifdef HAVE_VISIBILITY_ATTRIBUTE
__attribute__ ((visibility ("default")))
#endif
void R_init_bigoptim(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);

  M_R_cholmod_start(&c);
  c.final_ll = 1;	    /* LL' form of simplicial factorization */

  /* need own error handler, that resets  final_ll (after *_defaults()) : */
  c.error_handler = bigoptim_R_cholmod_error;
}

/** Finalizer for cplm called upon unloading the package.
 *
 */
void R_unload_bigoptim(DllInfo *dll) {
  M_cholmod_finish(&c);
}
