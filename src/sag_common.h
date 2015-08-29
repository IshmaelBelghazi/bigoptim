#ifndef SAG_COMMON_H_
#define SAG_COMMON_H_
/* R headers*/
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <R_ext/BLAS.h>
/* Spase Matrices support */
#include "Matrix.h"
#include "cholmod.h"
/* Utils */
#include "utils.h"
/* Shared variables */
#include "shared.h"
/* DEBUG   */
#ifndef DEBUG
#define DEBUG 1  // Enable/diasble traces.
#endif
/*  Types                 */
/* Loss functions pointer */
typedef double (*loss_fun)(double, double);
typedef double (*loss_grad_fun)(double, double);
/* Constant */
const static double precision = 1.490116119384765625e-8;
/* trainer type enum */
typedef enum {CONSTANT, LINESEARCH, ADAPTIVE} SAG_TYPE;
/* Prototypes */
/* Error Checking */
void validate_inputs(SEXP w, SEXP Xt, SEXP y, SEXP d, SEXP g, SEXP covered, SEXP sparse);
/* Monitor weights initialization */
SEXP initialize_monitor(SEXP monitor, SEXP maxIter, SEXP Xt);
#endif /* SAG_COMMON_H_ */
