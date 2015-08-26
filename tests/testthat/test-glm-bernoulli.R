context("GLM -- BERNOULLI")
#####################################################
## Testing Procedure                                # 
#####################################################
## A Empirical Data                                 #
#####################################################
## A.1 - approximate gradient is small on           # 
##       empirical data.                            #
## A.2 - Real Gradient is small on empirical data.  #
#####################################################
## B Simulated Data                                 #
#####################################################
## B.1- approximate gradient is small on            #
##       simulated data.                            #
## B.2 - Real Gradient is small on simulated data.  #
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
data(mini.covtype.libsvm)

empr_data <- list()
empr_data$y <- matrix(mini.covtype.libsvm$y, nrow=NROW(mini.covtype.libsvm$y), ncol=1)
empr_data$y[empr_data$y == 2] <- -1
empr_data$X <- cbind(rep(1, NROW(mini.covtype.libsvm$X)), mini.covtype.libsvm$X)
sample_size <- NROW(empr_data$X)
## Test parmeters
eps <- 1e-02
## Training parameters
tol <- 1e-4  ## Stop training when norm of approximate gradient is smaller than tol
lambda <- 1/sample_size
## A. Empirical Data tests
## Fitting empirical data with SAG
sag_empr_fits <- lapply(algs, function(alg) sag_fit(empr_data$X, empr_data$y,
                                                    lambda=lambda,
                                                    model=model,
                                                    standardize=FALSE,
                                                    tol=tol,
                                                    fit_alg=alg))

## A.1: Approximate gradient is small on simulated data
approx_grad_norm <- lapply(sag_empr_fits, function(fit) norm(get_approx_grad(fit), 'F'))

test_that("Approximate gradient is small on empirical data", {
  expect_less_than(approx_grad_norm$constant, eps)
  expect_less_than(approx_grad_norm$linesearch, eps)
  expect_less_than(approx_grad_norm$adaptive, eps)
})

## A.2: True gradient is small on simulated data
empr_grad <- lapply(sag_empr_fits, function(fit) {
  get_grad(fit, empr_data$X, empr_data$y)
})

empr_grad_norm <- lapply(empr_grad, function(grad) norm(grad, 'F'))

test_that("True Gradient is small on empirical data", {
  expect_less_than(empr_grad_norm$constant, eps)
  expect_less_than(empr_grad_norm$linesearch, eps)
  expect_less_than(empr_grad_norm$adaptive, eps)
})

## B. Simulated Data tests
## Generating simulated data
sample_size <- 3000
maxiter <- sample_size * 10
true_params <- c(1:3)
sim_data <- .simulate_binomial(true_params, sample_size=sample_size, intercept=FALSE)
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
approx_grad_norm <- lapply(sag_sim_fits, function(fit) norm(get_approx_grad(fit), 'F'))

test_that("Approximate gradient is small on simulated data", {
  expect_less_than(approx_grad_norm$constant, eps)
  expect_less_than(approx_grad_norm$linesearch, eps)
  expect_less_than(approx_grad_norm$adaptive, eps)
})

## B.2: True gradient is small on simulated data
sim_grad <- lapply(sag_sim_fits, function(fit) {
  get_grad(fit, sim_data$X, sim_data$y)
})

sim_grad_norm <- lapply(sim_grad, function(grad) norm(grad, 'F'))

test_that("True Gradient is small on simulated data", {
  expect_less_than(sim_grad_norm$constant, eps)
  expect_less_than(sim_grad_norm$linesearch, eps)
  expect_less_than(sim_grad_norm$adaptive, eps)
})
