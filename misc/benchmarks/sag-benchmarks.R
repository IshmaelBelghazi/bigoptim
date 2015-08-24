library(devtools)
library(microbenchmark)
load_all()
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
maxiter <- n * 10  ## 10 passes throught the dataset
lambda <- 1/n 
tol <- 0
print("starting benchmarks")
## -----------------------------------------------------------------------------
## SAG with Constant step size
print("Running Stochastic Average Gradient with constant step size")
sag_constant_time <- system.time({
  sag_constant_fit <- sag_fit(X=X, y=y, lambda=lambda, maxiter=maxiter,
                              tol=tol, family="binomial",
                              fit_alg="constant", standardize=FALSE)
})
print("sag constant time:")
print(sag_constant_time)

## -----------------------------------------------------------------------------
## SAG with linesearch
print("Running Stochastic Average Gradient with line-search")
Li <- 1
sag_linesearch_time <- system.time({
  sag_ls_fit <- sag_fit(X=X, y=y, lambda=lambda,
                        Li=Li, stepSizeType=1,
                        maxiter=maxiter, 
                        tol=tol, family="binomial", fit_alg="linesearch",
                        standardize=FALSE)
})
print("sag linesearch time:")
print(sag_linesearch_time)
## -----------------------------------------------------------------------------
## SAG with linesearch and adaptive sampling
print(paste0("Running Stochastic Average Gradient with ",
             "linesearch and adaptive sampling"))
sag_adaptive_time <- system.time({
  sag_adaptive_fit <- sag_fit(X, y, lambda=lambda,
                              maxiter=maxiter, 
                              tol=tol, family="binomial", fit_alg="adaptive",
                              standardize=FALSE)
})
print("sag adaptive time:")
print(sag_adaptive_time)
