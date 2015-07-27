rm(list=ls())
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(reshape2))
suppressPackageStartupMessages(library(glmnet))
family <- 1  ## 1 for Bernoulli
## Loading Data set
data(covtype.libsvm)
## Normalizing Columns and adding intercept
X <- cbind(rep(1, NROW(covtype.libsvm$X)), scale(covtype.libsvm$X))
y <- covtype.libsvm$y
y[y == 2] <- -1
## Setting seed
set.seed(0)
## Setting up problem
## Number of lambdas to compute
nlambda <- 5
## Getting lambdas grid
glmnet_fit <- glmnet(X, as.factor(y), alpha=0, family="binomial",
                     nlambda=nlambda, standardize=FALSE, intercept=FALSE)
lambdas <- rev(glmnet_fit$lambda)

trainers <- list(constant=list(fun=sag_constant,
                               params=list(covered=NULL,
                                           w=NULL,
                                           d=NULL,
                                           g=NULL)),
                 ls=list(fun=sag_ls,
                         params=list(covered=NULL,
                                     w=NULL,
                                     d=NULL,
                                     g=NULL)))

lambda <- lambdas[1]
monitor_training <- function(X, y, lambda, iVals,
                             trainers_list=trainers,
                             maxiter=NROW(X) * 10,
                             tol=0,
                             training_periods=100,
                             grad_fun=.bernoulli_cost_grad,
                             cost_fun=.bernoulli_cost_C,
                             verbose=TRUE, ...) {
  iVals <- matrix(sample.int(NROW(X), size=maxiter, replace=TRUE),
                  nrow=maxiter, ncol=1)
  ## Number of training period (number of intervals from 1 to maxiter)
  training_breaks <- c(seq(1, NROW(iVals) - NROW(iVals) %% training_periods,
                           by=NROW(iVals) %/% training_periods), NROW(iVals))
  training_table <- data.frame(iteration=numeric(),
                              grad_norm=numeric(),
                              cost=numeric(),
                              fit_alg=character())
  ## Training with warm starting.
  for (i in 1:training_periods) {
  if(verbose) {
    print(sprintf("Processing training period: %d/%d", i, training_periods))
  }
    iVals_i <- iVals[training_breaks[i]:training_breaks[i + 1],, drop=FALSE]
    ## Fitting model
    fits_i <- lapply(trainers, function(model) {
      fit <- model$fun(X, y, wInit=model$params$w, lambda=lambda,
                       iVals=iVals_i,
                       d=model$params$d,
                       g=model$params$g,
                       covered=model$params$covered,
                       tol=tol,
                       family=family)
      fit <- list(params=fit[names(model$params)])
      fit
    })
    trainers <- modifyList(trainers, fits_i, keep.null=TRUE)
    ## Computing gradient norm and loss
    results_i <- mapply(function(trainer, trainer_name) {
      data.frame(iteration=training_breaks[i + 1],
                 grad_norm=norm(grad_fun(X, y,
                                         trainer$params$w,
                                         lambda=lambda), 'F'),
                 cost=cost_fun(X, y,
                               trainer$params$w, lambda=lambda),
                 fit_alg=trainer_name)},
                 trainers, names(trainers),
                 USE.NAMES=FALSE,
                 SIMPLIFY=FALSE)
    training_table <- rbind(training_table, do.call(rbind, results_i))
  }
  ## Adding glmnet
  return(training_table)
}

training_table <- monitor_training(X, y, lambda=lambda, iVals=iVals,
                                   trainers_list=trainers,
                                   grad_fun=.bernoulli_cost_grad_C,
                                   cost_fun=.bernoulli_cost_C)
training_table$fit_alg <- as.factor(training_table$fit_alg)
## Plotting Tab graphs
cost_graph <- ggplot(training_table,
                     aes(x=iteration, y=cost, color=fit_alg)) +
  geom_line()

grad_norm_graph <- ggplot(training_table,
                          aes(x=iteration, y=grad_norm, color=fit_alg)) +
  geom_line()

