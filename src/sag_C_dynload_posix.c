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
  CHK_DL_OPEN(handle = dlopen(filename, RTLD_NOW));
  /* Assigning loss function */
  dlerror();  /* Clearing previous error */
  //container.dyn_loss_fun = (loss_fun) dlsym(handle, loss_symbol);
  //void * dyn_loss = dlsym(handle, loss_symbol);
  CHK_DL_LOAD(container.dyn_loss_fun = (loss_fun) dlsym(handle, loss_symbol));
  /* Assigning Gradient function */
  dlerror();  /* Clearing previous error */
  CHK_DL_LOAD(container.dyn_loss_grad_fun = (loss_grad_fun) dlsym(handle, grad_symbol));
  /* Assigning shared library handle */
  container.handle = handle;

  return container;
}


