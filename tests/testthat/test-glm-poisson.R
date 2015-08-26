context("GLM -- POISSON")
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
family <- "poisson"
algs <- list(##constant="constant",
             linesearch="linesearch",
             adaptive="adaptive")
## Data
## Empirical data
data(GlmnetExamples)
dataset <- list()
dataset$y <- GlmnetExamples$poisson$y
dataset$y <- matrix(as.double(dataset$y), nrow=NROW(dataset$y), ncol=1)
dataset$X <- cbind(rep(1, NROW(GlmnetExamples$poisson$X)),
                   scale(GlmnetExamples$poisson$X))
sample_size <- NROW(dataset$X)
## Test parmeters
eps <- 1e-02
## Training parameters
tol <- 1e-03  ## Stop training when norm of approximate gradient is smaller than tol
lambda <- 1/sample_size
## A. Empirical Data tests
sag_empr_fits <- lapply(algs, function(alg) sag_fit(dataset$X, dataset$y,
                                                    lambda=lambda,
                                                    family=family,
                                                    standardize=FALSE,
                                                    tol=tol,
                                                    fit_alg=alg))

## A.1: Approximate gradient is small on simulated data
approx_grad_norm <- lapply(sag_empr_fits, function(fit) norm(get_approx_grad(fit), 'F'))

test_that("Approximate gradient is small on empirical data", {
  ##expect_less_than(approx_grad_norm$constant, eps)
  expect_less_than(approx_grad_norm$linesearch, eps)
  expect_less_than(approx_grad_norm$adaptive, eps)
})

## A.2: True gradient is small on simulated data
empr_grad <- lapply(sag_empr_fits, function(fit) {
  get_grad(fit, dataset$X, dataset$y)
})

empr_grad_norm <- lapply(empr_grad, function(grad) norm(grad, 'F'))

test_that("True Gradient is small on empirical data", {
  ##expect_less_than(empr_grad_norm$constant, eps)
  expect_less_than(empr_grad_norm$linesearch, eps)
  expect_less_than(empr_grad_norm$adaptive, eps)
})
