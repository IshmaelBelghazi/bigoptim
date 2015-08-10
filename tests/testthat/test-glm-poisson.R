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
## ## Require libraries
## ## Algorithms
## model <- "poisson"
## algs <- list(constant="constant",
##              linesearch="linesearch")
## ## Data
## ## Empirical data
## data(GlmnetExamples)
## dataset <- list()
## dataset$y <- GlmnetExamples$poisson$y
## dataset$y <- matrix(as.double(dataset$y), nrow=NROW(dataset$y), ncol=1)
## dataset$X <- cbind(rep(1, NROW(GlmnetExamples$poisson$X)),
##                    scale(GlmnetExamples$poisson$X))
## sample_size <- NROW(dataset$X)
## ## Test parmeters
## eps <- 1e-02
## ## Training parameters
## tol <- 0.0  ## Stop training when norm of approximate gradient is smaller than tol
## maxiter <- sample_size * 10
## lambda <- 1/sample_size
## ## A. Empirical Data tests
## ## Subsetting empirical data
## empr_data <- dataset
## ## Fitting empirical data with SAG
## sag_empr_fits <- lapply(algs, function(alg) sag_fit(empr_data$X, empr_data$y,
##                                                     lambda=lambda,
##                                                     maxiter=maxiter,
##                                                     model=model,
##                                                     standardize=FALSE,
##                                                     tol=tol,
##                                                     fit_alg=alg))

## ## A.1: Approximate gradient is small on simulated data
## approx_grad_norm <- lapply(sag_empr_fits, function(fit) norm(fit$approx_grad, 'F'))

## test_that("Approximate gradient is small on empirical data", {
##   expect_less_than(approx_grad_norm$constant, eps)
##   expect_less_than(approx_grad_norm$linesearch, eps)
## })

## ## A.2: True gradient is small on simulated data
## empr_grad <- lapply(sag_empr_fits, function(fit) {
##   .binomial_cost_grad(empr_data$X,
##                   empr_data$y,
##                   coef(fit),
##                   lambda=lambda,
##                   backend="C")})

## empr_grad_norm <- lapply(empr_grad, function(grad) norm(grad, 'F'))

## test_that("True Gradient is small on empirical data", {
##   expect_less_than(empr_grad_norm$constant, eps)
##   expect_less_than(empr_grad_norm$linesearch, eps)
## })
