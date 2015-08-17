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
maxiter <- n * 10  ## 10 passes throught the dataset
lambda <- 1/n 
tol <- 0

## -----------------------------------------------------------------------------
## SAG with Constant step size
print("Running Stochastic Average Gradient with constant step size")
sag_constant_fit <- sag_fit(X=X, y=y, lambda=lambda, maxiter=maxiter,
                            family=family, tol=tol, model="binomial",
                            fit_alg="constant", standardize=FALSE)
cost_constant <- .binomial_cost(X, y, coef(sag_constant_fit), lambda=lambda, backend="C")
print(sprintf("Cost is: %f. Value in Mark's matlab code: 0.513607",
              cost_constant))
cost_grad_const <- .binomial_cost_grad(X, y, coef(sag_constant_fit), lambda=lambda, backend="C")
cost_grad_const_norm <- norm(cost_grad_const, 'F')
print(sprintf("Gradient norm: %f. Value in Mark's matlab code: 0.001394", cost_grad_const_norm))


## -----------------------------------------------------------------------------
## SAG with linesearch
print("Running Stochastic Average Gradient with line-search")
Lmax <- 1
sag_ls_fit <- sag_fit(X=X, y=y, lambda=lambda,
                      stepSize=Lmax, stepSizeType=1,
                      maxiter=maxiter, family=family,
                      tol=tol, model="binomial", fit_alg="linesearch",
                      standardize=FALSE)
cost_ls <- .binomial_cost(X, y, coef(sag_ls_fit), lambda=lambda, backend="C")
print(sprintf("Cost is: %f. Value in Mark's matlab code: 0.513497",
              cost_ls))
cost_grad_ls <- .binomial_cost_grad(X, y, coef(sag_ls_fit), lambda=lambda, backend="C")
cost_grad_ls_norm <- norm(cost_grad_ls, 'F')
print(sprintf("Gradient norm: %f. Value in Mark's matlab code: 0.001394", cost_grad_ls_norm))

## -----------------------------------------------------------------------------
## SAG with linesearch and adaptive sampling
print(paste0("Running Stoachastic Average Gradient with ",
             "linesearch and adaptive sampling"))
randVals <- matrix(runif(maxiter * 2), nrow=maxiter, ncol=2)
sag_adaptive_fit_mark <- sag_adaptive_ls(X, y, lambda=lambda, maxiter=maxiter,
                                          randVals=randVals, tol=tol, standardize=FALSE)
sag_adaptive_fit <- sag_fit(X, y, lambda=lambda, maxiter=maxiter, randVals=randVals,
                           tol=tol, model="binomial", fit_alg="adaptive", standardize=FALSE)


cost_adaptive <- .binomial_cost(X, y, sag_adaptive_fit$w, lambda=lambda, backend="C")
cost_adaptive_mark <- .binomial_cost(X, y, sag_adaptive_fit_mark$w, lambda=lambda, backend="C")

print(sprintf("Cost is: %f. Value in Mark's matlab code: TBD",
              cost_adaptive))

print(sprintf("(mark) Cost is: %f . Value in Mark's matlab code: TBD",
              cost_adaptive_mark))

cost_grad_adaptive <- .binomial_cost_grad(X, y, sag_adaptive_fit$w, lambda=lambda, backend="C")
cost_grad_adaptive_norm <- norm(cost_grad_adaptive, 'F')
print(sprintf("Gradient norm: %f. Value in Mark's matlab code: TBD", cost_grad_adaptive_norm))
cost_grad_adaptive_mark <- .binomial_cost_grad(X, y, sag_adaptive_fit_mark$w, lambda=lambda, backend="C")
cost_grad_adaptive_norm_mark <- norm(cost_grad_adaptive_mark, 'F')
print(sprintf("Gradient norm: %f. Value in Mark's matlab code: TBD", cost_grad_adaptive_norm_mark))
 
