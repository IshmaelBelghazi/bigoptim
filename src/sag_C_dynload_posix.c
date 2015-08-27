#include "sag_C_dynload_posix.h"

/* Dynamically load shared library object from path and returns
   pointer to function.
 */

/* Container for dynamically loaded loss function and handle */
dyn_fun_container load_C_shared_model(const char * filename,
                                      const char * loss_symbol,
                                      const char * grad_symbol) {
  void * handle;
  dyn_fun_container container;
  /* Opening passed shared object */
  /* Opening passed shared object */
  handle = dlopen(filename, RTLD_NOW);
  CHK_DL_OPEN(handle);

  /* Assigning loss function */
  container.dyn_loss_fun = (loss_fun) dlsym(handle, loss_symbol);
  CHK_DL_LOAD(container.dyn_loss_fun);
  /* Assigning Gradient function */
  container.dyn_loss_grad_fun = (loss_grad_fun) dlsym(handle, grad_symbol);
  CHK_DL_LOAD(container.dyn_loss_grad_fun);
  /* Assigning shared library handle */
  container.handle = handle;

  return container;
}


