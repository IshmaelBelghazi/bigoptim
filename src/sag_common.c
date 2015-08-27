#include "sag_common.h"

/* Sparse Matrices stubs */
#include "Matrix_stubs.c"

/* Error Checking */
void validate_inputs(SEXP w, SEXP Xt, SEXP y, SEXP d, SEXP g, SEXP covered, SEXP sparse) {

  int Xt_nrows; int Xt_ncols;
  if (*INTEGER(sparse)) {
    CHM_SP cXt = AS_CHM_SP(Xt);
    Xt_nrows = cXt->nrow;
    Xt_ncols = cXt->ncol;
  } else {
    Xt_nrows = INTEGER(GET_DIM(Xt))[0];
    Xt_ncols = INTEGER(GET_DIM(Xt))[1];
  }

  if ( Xt_nrows != INTEGER(GET_DIM(w))[0]) {
    error("w and Xt must have the same number of rows");
  }
  if (Xt_ncols != INTEGER(GET_DIM(y))[0]) {
    error("number of columns of Xt must be the same as the number of rows in y");
  }
  if ( INTEGER(GET_DIM(w))[0]!= INTEGER(GET_DIM(d))[0]) {
    error("w and d must have the same number of rows");
  }
  if (INTEGER(GET_DIM(y))[0] != INTEGER(GET_DIM(g))[0]) {
    error("y and g must have the same number of rows");
  }
  if (INTEGER(GET_DIM(covered))[0] != INTEGER(GET_DIM(y))[0]) {
    error("covered and y must have the same number of rows");
  }
}
/* Initialise Monitor */
SEXP initialize_monitor(SEXP monitor, SEXP maxIter, SEXP Xt) {
  SEXP monitor_w;
  if (*INTEGER(monitor)) {
    int nVars = INTEGER(GET_DIM(Xt))[0];
    int nPasses = *INTEGER(maxIter)/INTEGER(GET_DIM(Xt))[1];
    monitor_w = PROTECT(allocMatrix(REALSXP , nVars, nPasses + 1));
    Memzero(REAL(monitor_w), (nPasses + 1) * nVars);
  } else {
    monitor_w = R_NilValue;
  }
  UNPROTECT(1);
  return monitor_w;
}
