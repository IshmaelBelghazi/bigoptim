#ifndef SAG_C_DYNLOAD_POSIX_H_
#define SAG_C_DYNLOAD_POSIX_H_
#include "sag_common.h"
#include "dlfcn.h"

/* Dynamically loaded loss and gradient container */
typedef struct dyn_fun_container {
  loss_fun dyn_loss_fun;
  loss_grad_fun dyn_loss_grad_fun;
  void * handle;
} dyn_fun_container;

/* Error Handling Macros  */
/* Check dynamic library opening */
#define CHK_DL_OPEN( x )                                    \
dlerror();                                                  \
  if (!( x )) {                                             \
    error("Dynamic library operning error, %s", dlerror()); \
  }
/* Check dynamic library loading */
#define CHK_DL_LOAD( x )                                         \
  dlerror();                                                     \
  if (!( x )) {                                                  \
    error("Dynamic library loading error: %s", dlerror());       \
}
/* Load model function pointers from dynamically shared object */
dyn_fun_container load_C_shared_model(const char * filename,
                                      const char * loss_symbol,
                                      const char * grad_symbol);
#endif /* SAG_C_DYNLOAD_POSIX_H_ */
