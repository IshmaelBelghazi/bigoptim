#ifndef SAG_TRAIN_H_
#define SAG_TRAIN_H_
#include "sag_common.h"
#include "sag_constant.h"
#include "sag_linesearch.h"
#include "sag_adaptive.h"
#include "dataset.h"
#include "trainers.h"
#include "glm_models.h"


void train(GlmTrainer* trainer, GlmModel* model, Dataset* dataset);
SEXP make_return_list(GlmTrainer* trainer, GlmModel* model, Dataset* dataset);
void cleanup(GlmTrainer* trainer, GlmModel* model, Dataset* dataset);
#endif /* SAG_TRAIN_H_*/
