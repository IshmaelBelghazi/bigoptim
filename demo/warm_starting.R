suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(glmnet))
## Loading Data set
data(covtype.libsvm)
## Normalizing Columns and adding intercept
X <- cbind(rep(1, NROW(covtype.libsvm$X)), scale(covtype.libsvm$X))
y <- covtype.libsvm$y
y[y == 2] <- -1
n <- NROW(X)
p <- NCOL(X)
## Setting seed
#set.seed(0)
## Setting up problem
n_passes <- 50  ## number of passses trough the dataset
maxiter <- n * n_passes
lambda <- 1/n 
tol <- 0

family <- "binomial"
## Fitting with glmnet ---------------------------------------------------------
glmnet_fits <- glmnet(X, y, family=family, nlambda=10, standardize=FALSE, intercept=FALSE, alpha=0)
## Getting Lambdas
lambdas <- glmnet_fits$lambda
sag_fits <- sapply(sort(lambdas), function(lambda)
                                    coef(sag_fit(X, y, lambda=lambda, maxiter=maxiter, tol=0)))
sag_fits_warm <- t(sag(X, y, lambdas=lambdas, maxiter=maxiter, tol=0, model=family, fit_alg="adaptive")$lambda_w)
grad_norm_fun <- function(w, lambda) norm(.C_binomial_cost_grad(X, y, w, lambda), 'F')
sag_fits_grad_norm <- data.frame(lambda=numeric(), sag_fit=numeric(), sag_fit_warm=numeric())
for (i in 1:length(lambdas)) {
  lambda_i <- sort(lambdas)[i]
  sag_fits_grad_norm[i, 1] <- lambda_i
  ##sag_fits_grad_norm[i, 2] <- grad_norm_fun(sag_fits[i,, drop=FALSE], lambda_i)
  sag_fits_grad_norm[i, 3] <- grad_norm_fun(sag_fits_warm[i,, drop=FALSE], lambda_i)
}
