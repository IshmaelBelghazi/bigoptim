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
tol <- 1e-3
family <- "binomial"
## Fitting with glmnet ---------------------------------------------------------
glmnet_fits <- glmnet(X, y, family=family, nlambda=10,
                      standardize=FALSE, intercept=FALSE, alpha=0)
## Getting Lambdas
lambdas <- glmnet_fits$lambda
## Fitting with stochastic average gradient descent with warm starting ----------
sag_fits_warm <- sag(X, y, lambdas=lambdas, maxiter=maxiter,
                     tol=tol, family=family, fit_alg="constant")
## Getting costs ----------------------------------------------------------------
costs <- get_cost(sag_fits_warm, X, y)
print(get_cost(sag_fit, X, y))
print(costs)
## Getting Gradients ------------------------------------------------------------
grads <- get_grad(sag_fits_warm, X, y)
print(grads)
