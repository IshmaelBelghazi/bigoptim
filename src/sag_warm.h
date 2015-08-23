#ifndef SAG_WARM_H_
#define SAG_WARM_H_
#include "sag_common.h"
#include "sag_train.h"
#include "utils.h"

void sag_warm(GlmTrainer* trainer, GlmModel* model, Dataset* dataset,
              double* lambdas, int nLambdas, double * lambda_w);
#endif
