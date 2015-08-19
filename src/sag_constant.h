#ifndef SAG_CONSTANT_H_
#define SAG_CONSTANT_H_
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "utils.h"
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"

void _sag_constant(GlmTrainer * trainer, GlmModel * model, Dataset * dataset);

#endif /* SAG_CONSTANT_H_ */
