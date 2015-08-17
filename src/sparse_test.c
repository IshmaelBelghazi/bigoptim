#include "sparse_test.h"
#include "Matrix.h"
#include <cholmod.h>

SEXP C_sparse_test(SEXP X) {
  /* Getting dgCMatrix underlying struc */
  CHM_SP cX = AS_CHM_SP(X);
  Rprintf("ncol is %d\n", cX->ncol);
  Rprintf("nrow is %d\n", cX->nrow);
  Rprintf("dtype is %s\n", cX->dtype);
  return R_NilValue;
}
