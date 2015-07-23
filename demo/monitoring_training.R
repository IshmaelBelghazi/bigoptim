##rm(list=ls())
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
n <- NROW(X)
p <- NCOL(X)
## Setting seed
set.seed(0)
## Setting up problem
lambda <- 1/n
tol <- 0
maxiter <- n * 10
iVals <- matrix(sample.int(n, size=maxiter, replace=TRUE),
                nrow=maxiter, ncol=1)
training_periods <- 100
training_breaks <- c(seq(1, NROW(iVals) - NROW(iVals) %% training_periods,
                         by=NROW(iVals) %/% training_periods), NROW(iVals))
## SAG with constant step size
covered_constant <- matrix(0L, nrow=n, ncol=1)
d_constant <- matrix(0, nrow=p, ncol=1)
g_constant <- matrix(0, nrow=n, ncol=1)
w_constant <- matrix(0, nrow=p, ncol=1)
## SAG with linesearch
covered_ls <- matrix(0L, nrow=n, ncol=1)
d_ls <- matrix(0, nrow=p, ncol=1)
g_ls <- matrix(0, nrow=n, ncol=1)
w_ls <- matrix(0, nrow=p, ncol=1)
## Loss 
grad_norm_table  <- data.frame(training_period=integer(),
                               grad_norm_constant=numeric(),
                               grad_norm_ls=numeric())
cost_table <- data.frame(training_period=integer(),
                         cost_constant=numeric(),
                         cost_ls=numeric())
## Fitting with glmnet
glmnet_fit <- glmnet(X, as.factor(y), alpha=0, family="binomial",
                     nlambda=5, standardize=FALSE, intercept=FALSE)
lambdas <- rev(glmnet_fit$lambda)
tables <- list()
for (lambda in lambdas) {
  for (i in 1:training_periods) {
    print(sprintf("Processing training period: %d/%d", i, training_periods))
    iVals_i <- iVals[training_breaks[i]:training_breaks[i + 1],, drop=FALSE]
    print("Fitting SAG with constant step size")
    suppressWarnings(sag_constant_fit <- sag_constant(X, y, lambda=lambda,
                                                      maxiter=maxiter,
                                                      iVals=iVals_i,
                                                      wInit=w_constant,
                                                      d=d_constant,
                                                      g=g_constant,
                                                      covered=covered_constant,
                                                      family=family,
                                                      tol=tol))
    covered_constant <- sag_constant_fit$covered
    d_constant <- sag_constant_fit$d
    g_constant <- sag_constant_fit$g
    w_constant <- sag_constant_fit$w
    print("Fitting SAG with linesearch")
    suppressWarnings(sag_ls_fit <- sag_ls(X, y, lambda=lambda,
                                          maxiter=maxiter,
                                          iVals=iVals_i,
                                          wInit=w_ls,
                                          d=d_ls,
                                          g=g_ls,
                                          covered=covered_ls,
                                          family=family,
                                          tol=tol))

    covered_ls <- sag_ls_fit$covered
    d_ls <- sag_ls_fit$d
    g_ls <- sag_ls_fit$g
    w_ls <- sag_ls_fit$w

    cost_table[i, 'training_period'] <- i
    cost_table[i, 'cost_constant'] <- .bernoulli_loss(X, y, w_constant, lambda=lambda)
    cost_table[i, 'cost_ls'] <- .bernoulli_loss(X, y, w_ls, lambda=lambda)
    grad_norm_table[i, 'training_period'] <- i
    grad_norm_table[i, 'grad_norm_constant'] <- norm(.bernoulli_grad(X, y,
                                                                     w_constant, lambda=lambda), 'F')
    grad_norm_table[i, 'grad_norm_ls'] <- norm(.bernoulli_grad(X, y, w_ls, lambda=lambda), 'F')
  }
  tables[[as.character(lambda)]] <- list(grad_norm=grad_norm_table,
                                       cost=cost_table)
}



