#include <R.h>
#include "Matrix.h"
/* ENTRY POINTS -- START */
#include "entrypoint-glm.h"
#include "entrypoint-sag_constant.h"
#include "entrypoint-sag_linesearch.h"
#include "entrypoint-sag_adaptive.h"
/* ENTRY POINTS -- END */

/* TEMPORARY -- START */
#include "sparse_test.h"
#include "sag_constant_marks.h"
/* TEMPORTARY -- END */
#include <R_ext/Rdynload.h>

/** utitlity macro in registering native routines */
#define CALLDEF(name, n)  {#name, (DL_FUNC) &name, n}

/** function that registers native routines  */
static R_CallMethodDef CallEntries[] = {

  CALLDEF(C_binomial_cost, 4),
  CALLDEF(C_binomial_cost_grad, 4),
  CALLDEF(C_gaussian_cost, 4),
  CALLDEF(C_gaussian_cost_grad, 4),
  CALLDEF(C_exponential_cost, 4),
  CALLDEF(C_exponential_cost_grad, 4),
  CALLDEF(C_poisson_cost, 4),
  CALLDEF(C_poisson_cost_grad, 4),
  CALLDEF(C_sag_constant, 12),
  CALLDEF(C_sag_linesearch, 13),
  CALLDEF(C_sag_adaptive, 14),
  /* TEMPORARY -- STARTS*/
  CALLDEF(C_sparse_test, 1),
  CALLDEF(C_sag_constant_mark, 9),
  /* TEMPORARY -- ENDS */
  {NULL, NULL, 0}
};

/* local cholmod_common struct*/
cholmod_common c;

/** This is the CHOLMOD error handler from lme4*/
void attribute_hidden bigoptim_R_cholmod_error(int status,
                                               const char *file,
                                               int line,
                                               const char *message) {
  if(status < 0) {
#ifdef Failure_in_matching_Matrix
    /* This fails unexpectedly with
     *  function 'cholmod_l_defaults' not provided by package 'Matrix'
     * from ../tests/lmer-1.R 's  (l.68)  lmer(y ~ habitat + (1|habitat*lagoon)
     */
    M_cholmod_defaults(&c);/* <--- restore defaults,
                            * as we will not be able to .. */
    c.final_ll = 1;	    /* LL' form of simplicial factorization */
#endif

    error("Cholmod error '%s' at file:%s, line %d", message, file, line);
  }
  else
    warning("Cholmod warning '%s' at file:%s, line %d",
            message, file, line);
}
/** Initializer for cplm, called upon loading the package.
 *
 *  Initialize CHOLMOD and require the LL' form of the factorization.
 *  Install the symbols to be u sed by functions in the package.
 */
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
