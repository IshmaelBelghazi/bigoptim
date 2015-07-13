#ifndef SAG_STEP_H_
#define SAG_STEP_H_


#include <R.h>
#include <R_ext/BLAS.h>
#include <math.h>
#include "glm_models.h"
#include "dataset.h"
#include "trainers.h"
#include "utils.h"

void _sag_constant_iteration(GlmTrainer * trainer, GlmModel * model, Dataset * dataset);
void _sag_linesearch_iteration(GlmTrainer * trainer, GlmModel * model, Dataset * dataset);
void _sag_adaptive_iteration(GlmTrainer * trainer, GlmModel * model, Dataset * dataset);


#endif /* SAG_STEP_H_ */
