context("GLM -- BERNOULLI")
#####################################################
## Testing Procedure                                # 
#####################################################
## A Simulated Data                                 #
#####################################################
## A.1 - True simulations parameters                #
##       are recovered on simulated data.           #
## A.2 - Consitency with glmnet on simulated data.  # 
## A.3 - approximate gradient is small on           #
##       simulated data.                            #
## A.4 - Real Gradient is small on simulated data.  #
#####################################################
## B Empirical Data                                 #
#####################################################
## B.1 - Consitency with glmnet on empirical data.  # 
## B.2 - approximate gradient is small on           # 
##       empirical data.                            #
## B.3 - Real Gradient is small on empirical data.  #
#####################################################

######################
## Setting up Tests ##
######################
## Require libraries
suppressMessages(library(glmnet))
## Algorithms
SAG <- list(constant=sag_constant, linesearch=sag_ls)
family <- 1  ## 0 for gaussian, 1 for Bernoulli, 2 for exponential and 3 for poisson
## Data
sample_size <- 10000  ## For sampled and empirical data
## Sampled Data
true_params <- matrix(c(1, 2, 3), ncol=1)
sim_data <- .simulate_bernoulli(true_params, sample_size, intercept=FALSE)
## Empirical data
data(covtype.libsvm)
## Test parmeters
eps <- 1e-02
## Training parameters
tol <- 1e-8  ## Stop training when norm of approximate gradient is smaller than tol
maxIter <- sample_size * 200


## A. Simulated Data tests
## Fitting simulated data with SAG
sag_sim_fits <- lapply(SAG, function(alg) alg(sim_data$X,
                                              sim_data$y,
                                              lambda=0,
                                              maxiter=maxIter,
                                              tol=tol,
                                              family=family))
## Fitting with glmnet
glm_fit <- glmnet(sim_data$X, as.factor(sim_data$y), family="binomial", intercept=FALSE,
                  lambda=0)
glmnet_sim_hat <- as.matrix(coef(glm_fit))[-1, , drop=FALSE]
colnames(glmnet_sim_hat) <- rownames(glmnet_sim_hat) <- NULL  ## Fitting simulated data with glmnet
## A.1: Real Parameters are recovered
test_that("SAG algorithms recover true parameters on simulated data", {
  expect_equal(sag_sim_fits$constant$w, true_params, scale=1, tolerance=0.1)
  expect_equal(sag_sim_fits$linesearch$w, true_params, scale=1, tolerance=0.1)
})
## A.2: Consistency with glmnet on simulated data
test_that("SAG and glmnet results are consistent on simulated data", {
  expect_less_than(norm(sag_sim_fits$constant$w - glmnet_sim_hat, 'F'), eps)
  expect_less_than(norm(sag_sim_fits$linesearch$w - glmnet_sim_hat, 'F'), eps)
})
## A.3: Approximate gradient is small on simulated data
test_that("Approximate gradient is small on simulated data", {
  expect_less_than(norm(sag_sim_fits$constant$d, 'F'), tol)
  expect_less_than(norm(sag_sim_fits$linesearch$d, 'F'), tol)
})
## A.4: True gradient is small on simulated data
test_that("True Gradient is small on simulated data", {
  expect_less_than(mean(.bernoulli_grad(sim_data$X,
                                       sim_data$y,
                                       sag_sim_fits$constant$w,
                                       lambda=0)^2), 0.1)
  
  expect_less_than(mean(.bernoulli_grad(sim_data$X,
                                       sim_data$y,
                                       sag_sim_fits$linesearch$w,
                                       lambda=0)^2), 0.1)
})

## ##  B. Empirical Data tests
## ## Subsetting empirical data
## empr_data <- lapply(covtype.libsvm, function(X) X[1:sample_size, , drop=FALSE])
## empr_data$y[empr_data$y == 1] <- -1
## empr_data$y[empr_data$y == 2] <- 1
## ## Fitting empirical data with SAG
## sag_empr_fits <- lapply(SAG, function(alg) alg(empr_data$X,
##                                               empr_data$y,
##                                               lambda=0,
##                                               maxiter=maxIter,
##                                               tol=tol,
##                                               family=family))
## ## Fitting with glmnet
## glm_empr_fit <- glmnet(empr_data$X, as.factor(empr_data$y), family="binomial", intercept=FALSE,
##                        lambda=0)
## glmnet_empr_hat <- as.matrix(coef(glm_empr_fit))[-1, , drop=FALSE]
## colnames(glmnet_empr_hat) <- rownames(glmnet_empr_hat) <- NULL
## ## B.1: Consistency with glmnet on empirical data
## test_that("SAG and glmnet results are consistent on empirical data", {
##   expect_less_than(norm(sag_empr_fits$constant$w - glmnet_empr_hat, 'F'), eps)
##   expect_less_than(norm(sag_empr_fits$linesearch$w - glmnet_empr_hat, 'F'), eps)
## })
## ## B.2: Approximate gradient is small on simulated data
## test_that("Approximate gradient is small on empirical data" {
##   expect_less_than(norm(sag_empr_fits$constant$d, 'F'), tol)
##   expect_less_than(norm(sag_empr_fits$linesearch$d, 'F'), tol)
## })
## ## B.3: True gradient is small on simulated data
## test_that("True Gradient is small on empirical data", {
##   expect_less_than(sum(.bernoulli_grad(empr_data$X,
##                                        empr_data$y,
##                                        sag_empr_fits$constant$w,
##                                        lambda=0)^2), 0.1)
  
##   expect_less_than(sum(.bernoulli_grad(empr_data$X,
##                                        empr_data$y,
##                                        sag_empr_fits$linesearch$w,
##                                        lambda=0)^2), 0.1)
## })

