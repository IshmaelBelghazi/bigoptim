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
model <- "gaussian"
algs <- list(constant="constant", linesearch="linesearch")
## Data
sample_size <- 3000  ## For sampled and empirical data
## Empirical data
data(covtype.libsvm)
## Test parmeters
eps <- 1e-02
## Training parameters
tol <- 0  ## Stop training when norm of approximate gradient is smaller than tol
maxiter <- sample_size * 10
lambda <- 1/sample_size
## ## A. Empirical Data tests
## ## Subsetting empirical data
## empr_data <- lapply(covtype.libsvm, function(X) X[1:sample_size, , drop=FALSE])
## empr_data$y[empr_data$y == 1] <- -1
## empr_data$y[empr_data$y == 2] <- 1

## ## Fitting empirical data with SAG
## ## Constant SAG fit
## sag_empr_fits <- list()
## sag_empr_fits$constant <- sag_constant(empr_data$X,
##                                        empr_data$y,
##                                        lambda=lambda,
##                                        maxiter=maxIter,
##                                        tol=tol,
##                                        family=family)
## ## Linesearh SAG fit
## sag_empr_fits$linesearch <- sag_ls(empr_data$X,
##                                    empr_data$y,
##                                    lambda=lambda,
##                                    maxiter=maxIter,
##                                    tol=tol,
##                                    family=family)
## ## Glmnet empirical fit
## glmnet_empr_fit <- glmnet(empr_data$X,
##                           empr_data$y,
##                           family="binomial",
##                           intercept=FALSE,
##                           alpha=0,
##                           lambda=lambda)
## glmnet_empr_hat <- as.matrix(coef(glmnet_empr_fit))[-1, , drop=FALSE]
## colnames(glmnet_empr_hat) <- rownames(glmnet_empr_hat) <- NULL
## ## A.1: Consistency with glmnet on empirical data
## glmnet_SAG_cst_diff_norm <- norm(sag_empr_fits$constant$w - glmnet_empr_hat, 'F') 
## glmnet_SAG_ls_diff_norm <- norm(sag_empr_fits$linesearch$w - glmnet_empr_hat, 'F')
## test_that("SAG and glmnet results are consistent on empirical data", {
##   expect_less_than(glmnet_SAG_cst_diff_norm, eps)
##   expect_less_than(glmnet_SAG_ls_diff_norm, eps)
## })
## ## A.2: Approximate gradient is small on simulated data
## approx_grad_norm_constant <- anorm(sag_empr_fits$constant$g)
## approx_grad_norm_ls <- anorm(sag_empr_fits$linesearch$g)  
## test_that("Approximate gradient is small on empirical data", {
##   expect_less_than(approx_grad_norm_constant, tol)
##   expect_less_than(approx_grad_norm_ls, tol)
## })
## ## A.3: True gradient is small on simulated data
## empr_grad_constant <- .bernoulli_grad(empr_data$X,
##                                       empr_data$y,
##                                       sag_empr_fits$constant$w,
##                                       lambda=0)
## empr_grad_norm_constant <- anorm(empr_grad_constant)
## empr_grad_ls <- .bernoulli_grad(empr_data$X,
##                                         empr_data$y,
##                                         sag_empr_fits$linesearch$w,
##                                         lambda=0)
## empr_grad_norm_ls <- anorm(empr_grad_ls)
## test_that("True Gradient is small on empirical data", {
##   expect_less_than(empr_grad_norm_constant, 0.1)
##   expect_less_than(empr_grad_norm_ls, 0.1)
## })

## B. Simulated Data tests
## Generating simulated data
sample_size <- 3000
true_params <- c(1:3)
sim_data <- .simulate_gaussian(true_params, sample_size=sample_size, intercept=FALSE)
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

test_that("Approximate gradient is small on empirical data", {
  expect_less_than(approx_grad_norm$constant, eps)
  expect_less_than(approx_grad_norm$linesearch, eps)
})

## B.2: True gradient is small on simulated data
sim_grad <- lapply(sag_sim_fits, function(fit) {
  .binomial_cost_grad(sim_data$X,
                  sim_data$y,
                  coef(fit),
                  lambda=lambda,
                  backend="C")})

sim_grad_norm <- lapply(sim_grad, function(grad) norm(grad, 'F'))

test_that("True Gradient is small on empirical data", {
  expect_less_than(sim_grad_norm$constant, eps)
  expect_less_than(sim_grad_norm$linesearch, eps)
})
