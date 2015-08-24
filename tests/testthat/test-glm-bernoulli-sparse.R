context("GLM -- BERNOULLI - SPARSE")
#####################################################
## Testing Procedure                                # 
#####################################################
## A Empirical Data                                 #
#####################################################
## A.1 - approximate gradient is small on           # 
##       empirical data.                            #
## A.2 - Real Gradient is small on empirical data.  #
#####################################################
######################
## Setting up Tests ##
######################
##Require libraries
##Algorithms
model <- "binomial"
algs <- list(constant="constant",
             linesearch="linesearch",
             adaptive="adaptive")
## Data
## Empirical data
data(rcv1_train)
dataset <- list()
dataset$y <- rcv1_train$y
dataset$X <- rcv1_train$X
sample_size <- NROW(dataset$X)
## Test parmeters
eps <- 1e-02
## Training parameters
tol <- 1e-04  ## Stop training when norm of approximate gradient is smaller than tol
maxiter <- sample_size * 20
lambda <- 1/sample_size
## A. Empirical Data tests
## Fitting empirical data with SAG
sag_empr_fits <- lapply(algs, function(alg) sag_fit(dataset$X, dataset$y,
                                                    lambda=lambda,
                                                    maxiter=maxiter,
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

##A.2: True gradient is small on simulated data
empr_grad <- lapply(sag_empr_fits, function(fit) {
  get_grad(fit, dataset$X, dataset$y)
})

empr_grad_norm <- lapply(empr_grad, function(grad) norm(grad, 'F'))

test_that("True Gradient is small on empirical data", {
  expect_less_than(empr_grad_norm$constant, eps)
  expect_less_than(empr_grad_norm$linesearch, eps)
  expect_less_than(empr_grad_norm$adaptive, eps)
})
