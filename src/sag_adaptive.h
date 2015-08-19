#ifndef SAG_ADAPTIVE_H_
#define SAG_ADAPTIVE_H_
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include "glm_models.h"
#include "dataset.h"
#include "trainers.h"
#include "utils.h"

void _sag_adaptive(GlmTrainer *trainer, GlmModel *model, Dataset *dataset);

#endif /* SAG_ADAPTIVE_H_ */
