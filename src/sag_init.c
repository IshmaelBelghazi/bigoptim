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
  CALLDEF(C_sag_fit, 19),
  /* SAG with warm-starting */
  CALLDEF(C_sag, 18),
  {NULL, NULL, 0}
};

#ifdef HAVE_VISIBILITY_ATTRIBUTE
__attribute__ ((visibility ("default")))
#endif
void R_init_bigoptim(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
