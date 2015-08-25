suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(glmnet))
family <- "binomial"

fit_algs <- list(constant="constant",
                 linesearch="linesearch",
                 adaptive="adaptive")

time_fit <- function(X, y, lambda, maxiter, family, tol){
  fits <- lapply(fit_algs, function(fit_alg) {
    print(sprintf("processing: %s", fit_alg))
    fit_time <- system.time({fit <- sag_fit(X, y,
                                            lambda=lambda,
                                            maxiter=maxiter,
                                            family=family,
                                            fit_alg=fit_alg,
                                            standardize=FALSE,
                                            tol=tol)})
    list(cost=get_cost(fit, X, y),
         grad_norm=norm(get_grad(fit, X, y), 'F'),
         approx_grad_norm=norm(get_approx_grad(fit), 'F'),
         time=fit_time[['elapsed']])
  })
  print("processing glmnet")
  glmnet_time <- system.time({
    glmnet_fit <- glmnet(X, as.factor(y), alpha=0, family="binomial",
                         lambda=lambda, standardize=FALSE, intercept=FALSE)
  })
  if (is.sparse(X)) {
    backend <- "R"
    glmnet_w <- tail(coef(glmnet_fit), -1)
  } else {
    backend <- "C"
    glmnet_w <- as.matrix(coef(glmnet_fit))[-1]
  } 
  glmnet_w <- as.matrix(coef(glmnet_fit))[-1]
  backend <- if (is.sparse(X)) "R" else "C"
  glm_cost <- .get_cost(X, y, glmnet_w, lambda=lambda,
                        family=family, backend=backend)
  glm_cost_grad <- norm(.get_grad(X, y, glmnet_w, lambda=lambda,
                                  family=family, backend=backend), 'F')
  fits$glmnet <- list(cost=glm_cost,
                      grad_norm=glm_cost_grad,
                      approx_grad_norm=NULL,
                      time=glmnet_time[['elapsed']])
  do.call(cbind, fits)
}
## Covtype libsvm --------------------------------------------------------------
## Loading Data set
data(covtype.libsvm)
## Normalizing Columns and adding intercept
X_covtype <- cbind(rep(1, NROW(covtype.libsvm$X)), scale(covtype.libsvm$X))
y_covtype <- covtype.libsvm$y
y_covtype[y_covtype == 2] <- -1
## Setting seed
maxiter <- NROW(X_covtype) * 10 
lambda <- 1/NROW(X_covtype)
tol <- 1e-3
print("Timing on covertype ...")
fit_times_covtype <- time_fit(X_covtype, y_covtype, lambda, maxiter, family, tol)
print("... timing completed")
print(fit_times_covtype)
## Rcv1 train ------------------------------------------------------------------
data(rcv1_train)
X_rcv1_train <- rcv1_train$X
y_rcv1_train <- rcv1_train$y
maxiter <- NROW(X_covtype) * 5
lambda <- 1/NROW(X_covtype)
tol <- 0
print("Timing on covertype ...")
fit_times_rcv1_train <- time_fit(X_rcv1_train, y_rcv1_train,
                                 lambda, maxiter, family, tol)
print("... timing completed")
print(fit_times_rcv1_train)
