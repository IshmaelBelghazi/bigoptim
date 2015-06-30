#ifndef SAG_STEP_H_
#define SAG_STEP_H_

#include <R_ext/BLAS.h>
#include "glm_models.h"
#include "dataset.h"
#include "trainers.h"

void _sag_constant_iteration(GlmTrainer * trainer, GlmModel * model, Dataset * dataset);

#endif /* SAG_STEP_H_ */
