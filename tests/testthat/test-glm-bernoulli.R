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
family <- 1  ## 0 for gaussian, 1 for Bernoulli, 2 for exponential and 3 for poisson
## Data
## Empirical data
data(covtype.libsvm)
dataset <- list()
dataset$y <- covtype.libsvm$y
dataset$y[dataset$y ==  2] <- -1
dataset$X <- cbind(rep(1, NROW(covtype.libsvm$X)), scale(covtype.libsvm$X))
sample_size <- NROW(dataset$X)
## Test parmeters
eps <- 1e-02
## Training parameters
tol <- 0  ## Stop training when norm of approximate gradient is smaller than tol
maxIter <- sample_size * 10
lambda <- 1/sample_size
## A. Empirical Data tests
## Subsetting empirical data
empr_data <- dataset

## Fitting empirical data with SAG
## Constant SAG fit
sag_empr_fits <- list()
sag_empr_fits$constant <- sag_constant(empr_data$X,
                                       empr_data$y,
                                       lambda=lambda,
                                       maxiter=maxIter,
                                       tol=tol,
                                       family=family)
## Linesearh SAG fit
sag_empr_fits$linesearch <- sag_ls(empr_data$X,
                                   empr_data$y,
                                   lambda=lambda,
                                   maxiter=maxIter,
                                   tol=tol,
                                   family=family)
## A.1: Approximate gradient is small on simulated data
approx_grad_norm_constant <- abs(mean(sag_empr_fits$constant$g))
approx_grad_norm_ls <- abs(mean(sag_empr_fits$linesearch$g))
test_that("Approximate gradient is small on empirical data", {
  expect_less_than(approx_grad_norm_constant, eps)
  expect_less_than(approx_grad_norm_ls, eps)
})
## A.3: True gradient is small on simulated data
empr_grad_constant <- .bernoulli_grad(empr_data$X,
                                      empr_data$y,
                                      sag_empr_fits$constant$w,
                                      lambda=lambda)
empr_grad_norm_constant <- norm(empr_grad_constant, 'F')
empr_grad_ls <- .bernoulli_grad(empr_data$X,
                                empr_data$y,
                                sag_empr_fits$linesearch$w,
                                lambda=lambda)
empr_grad_norm_ls <- norm(empr_grad_ls, 'F')
test_that("True Gradient is small on empirical data", {
  expect_less_than(empr_grad_norm_constant, eps)
  expect_less_than(empr_grad_norm_ls, eps)
})

## B. Simulated Data tests
## Generating simulated data
sample_size <- 3000
true_params <- c(1:3)
sim_data <- .simulate_bernoulli(true_params, sample_size=sample_size, intercept=FALSE)
sim_data$X <- scale(sim_data$X)
## Fitting simulated data with SAG
## Constant SAG fit
sag_sim_fits <- list()
sag_sim_fits$constant <- sag_constant(sim_data$X,
                                       sim_data$y,
                                       lambda=lambda,
                                       maxiter=maxIter,
                                       tol=tol,
                                       family=family)
## Linesearh SAG fit
sag_sim_fits$linesearch <- sag_ls(sim_data$X,
                                  sim_data$y,
                                  lambda=lambda,
                                  maxiter=maxIter,
                                  tol=tol,
                                  family=family)
## B.1: Approximate gradient is small on simulated data
approx_grad_norm_constant <- abs(mean(sag_sim_fits$constant$g))
approx_grad_norm_ls <- abs(mean(sag_sim_fits$linesearch$g))  
test_that("Approximate gradient is small on simulated data", {
  expect_less_than(approx_grad_norm_constant, tol)
  expect_less_than(approx_grad_norm_ls, tol)
})
## B.2: True gradient is small on simulated data
sim_grad_constant <- .bernoulli_grad(sim_data$X,
                                      sim_data$y,
                                      sag_sim_fits$constant$w,
                                      lambda=lambda)
sim_grad_norm_constant <- norm(sim_grad_constant, 'F')
sim_grad_ls <- .bernoulli_grad(sim_data$X,
                               sim_data$y,
                               sag_sim_fits$linesearch$w,
                               lambda=lambda)
sim_grad_norm_ls <- norm(sim_grad_ls, 'F')
test_that("True Gradient is small on simulated data", {
  expect_less_than(sim_grad_norm_constant, eps)
  expect_less_than(sim_grad_norm_ls, eps)
})
