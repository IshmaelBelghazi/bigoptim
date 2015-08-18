#include "sparse_test.h"
#include "Matrix.h"
#include "cholmod.h"

SEXP C_sparse_test(SEXP X) {
  /* Getting dgCMatrix underlying struc */
  /* Rprintf("Printing from C\n"); */
  /* PrintValue(X); */
  CHM_SP cX = AS_CHM_SP(X);
  Rprintf("ncol is %d\n", cX->ncol);
  Rprintf("nrow is %d\n", cX->nrow);
  Rprintf("dtype is %s\n", cX->dtype);
  Rprintf("nzmax is %d\n", cX->nzmax);
  int * ir = cX->i;
  int * jc = cX->p;
  double * x = cX->x;

  Rprintf("Testing ir\n");
  for (int k = 0; k < (int)cX->nzmax; k++) {
    Rprintf("ir[%d]=%d\n", k, ir[k]);
  }
  Rprintf("Testing jc\n");
  for (int k = 0; k < (int)(cX->ncol + 1); k++) {
    Rprintf("jc[%d]=%d\n", k, jc[k]);
  }
  Rprintf("Testing x\n");
  for (int k=0; k < (int) cX->nzmax; k++){
    Rprintf("x: %f\n", x[k]);
  }
  return R_NilValue;
}
