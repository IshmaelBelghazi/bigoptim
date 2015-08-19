context("GLM -- BERNOULLI")
#####################################################
## Testing Procedure                                # 
#####################################################
## A Empirical Data                                 #
#####################################################
## A.1 - approximate gradient is small on           # 
##       empirical data.                            #
## A.1 - Real Gradient is small on empirical data.  #
#####################################################
## B Simulated Data                                 #
#####################################################
## B.1- approximate gradient is small on            #
##       simulated data.                            #
## B.1 - Real Gradient is small on simulated data.  #
#####################################################
######################
## Setting up Tests ##
######################
## Require libraries
## Algorithms
model <- "binomial"
algs <- list(constant="constant",
             linesearch="linesearch",
             adaptive="adaptive")
## Data
## Empirical data
data(covtype.libsvm)
dataset <- list()
dataset$y <- matrix(covtype.libsvm$y, nrow=NROW(covtype.libsvm$y), ncol=1)
dataset$y[dataset$y == 2] <- -1
dataset$X <- cbind(rep(1, NROW(covtype.libsvm$X)), scale(covtype.libsvm$X))
sample_size <- NROW(dataset$X)
## Test parmeters
eps <- 1e-02
## Training parameters
tol <- 0.00001  ## Stop training when norm of approximate gradient is smaller than tol
maxiter <- sample_size * 10
lambda <- 1/sample_size
## A. Empirical Data tests
## Subsetting empirical data
empr_data <- dataset
## Fitting empirical data with SAG
sag_empr_fits <- lapply(algs, function(alg) sag_fit(empr_data$X, empr_data$y,
                                                    lambda=lambda,
                                                    maxiter=maxiter,
                                                    model=model,
                                                    standardize=FALSE,
                                                    tol=tol,
                                                    fit_alg=alg))

## A.1: Approximate gradient is small on simulated data
approx_grad_norm <- lapply(sag_empr_fits, function(fit) norm(fit$approx_grad, 'F'))

test_that("Approximate gradient is small on empirical data", {
  expect_less_than(approx_grad_norm$constant, eps)
  expect_less_than(approx_grad_norm$linesearch, eps)
  expect_less_than(approx_grad_norm$adaptive, eps)
})

## A.2: True gradient is small on simulated data
empr_grad <- lapply(sag_empr_fits, function(fit) {
  .binomial_cost_grad(empr_data$X,
                  empr_data$y,
                  coef(fit),
                  lambda=lambda,
                  backend="C")})

empr_grad_norm <- lapply(empr_grad, function(grad) norm(grad, 'F'))

test_that("True Gradient is small on empirical data", {
  expect_less_than(empr_grad_norm$constant, eps)
  expect_less_than(empr_grad_norm$linesearch, eps)
  expect_less_than(empr_grad_norm$adaptive, eps)
})

## B. Simulated Data tests
## Generating simulated data
sample_size <- 3000
true_params <- c(1:3)
sim_data <- .simulate_bernoulli(true_params, sample_size=sample_size, intercept=FALSE)
sim_data$X <- scale(sim_data$X)
## Fitting simulated data with SAG
sag_sim_fits <- lapply(algs, function(alg) sag_fit(sim_data$X, sim_data$y,
                                                    lambda=lambda,
                                                    maxiter=maxiter,
                                                    model=model,
                                                    standardize=FALSE,
                                                    tol=tol,
                                                    fit_alg=alg))

## B.1: Approximate gradient is small on simulated data
approx_grad_norm <- lapply(sag_sim_fits, function(fit) norm(fit$approx_grad, 'F'))

test_that("Approximate gradient is small on simulated data", {
  expect_less_than(approx_grad_norm$constant, eps)
  expect_less_than(approx_grad_norm$linesearch, eps)
  expect_less_than(approx_grad_norm$adaptive, eps)
})

## B.2: True gradient is small on simulated data
sim_grad <- lapply(sag_sim_fits, function(fit) {
  .binomial_cost_grad(sim_data$X,
                  sim_data$y,
                  coef(fit),
                  lambda=lambda,
                  backend="C")})

sim_grad_norm <- lapply(sim_grad, function(grad) norm(grad, 'F'))

test_that("True Gradient is small on simulated data", {
  expect_less_than(sim_grad_norm$constant, eps)
  expect_less_than(sim_grad_norm$linesearch, eps)
  expect_less_than(sim_grad_norm$adaptive, eps)
})
