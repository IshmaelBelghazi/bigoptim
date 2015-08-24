context("GLM -- GAUSSIAN")
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
## Algorithms
family <- "gaussian"
algs <- list(##constant="constant",
             linesearch="linesearch",
             adaptive="adaptive")
## Data
sample_size <- 3000  ## For sampled and empirical data
## Test parmeters
eps <- 1e-02
## Training parameters
tol <- 0 ## Stop training when norm of approximate gradient is smaller than tol
maxiter <- sample_size * 50 
lambda <- 1/sample_size
## B. Simulated Data tests
## Generating simulated data
true_params <- c(1:3)
sim_data <- .simulate_gaussian(true_params, sample_size=sample_size,
                               intercept=FALSE)
sim_data$X <- scale(sim_data$X)
## Fitting simulated data with SAG
sag_sim_fits <- lapply(algs, function(alg) sag_fit(sim_data$X, sim_data$y,
                                                   lambda=lambda,
                                                   maxiter=maxiter,
                                                   family=family,
                                                   standardize=FALSE,
                                                   tol=tol,
                                                   fit_alg=alg))

## B.1: Approximate gradient is small on simulated data
approx_grad_norm <- lapply(sag_sim_fits, function(fit)
                                           norm(get_approx_grad(fit), 'F'))

test_that("Approximate gradient is small on simulated data", {
  ##expect_less_than(approx_grad_norm$constant, eps)
  expect_less_than(approx_grad_norm$linesearch, eps)
  expect_less_than(approx_grad_norm$adaptive, eps)
})

## B.2: True gradient is small on simulated data
sim_grad <- lapply(sag_sim_fits, function(fit) {
  get_grad(fit, sim_data$X, sim_data$y)
})
sim_grad_norm <- lapply(sim_grad, function(grad) norm(grad, 'F'))

test_that("True Gradient is small on simulated data", {
  ##expect_less_than(sim_grad_norm$constant, eps)
  expect_less_than(sim_grad_norm$linesearch, eps)
  expect_less_than(sim_grad_norm$adaptive, eps)
})
