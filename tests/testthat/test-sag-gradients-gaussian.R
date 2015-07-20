context("Gradient tests -- GAUSSIAN")
## test parameters
eps <- 1e-02

## Simulating logistic datasets
true_params <- c(1, 2, 3)
sample_size <- 10000
lambda <- 0
sim <- .simulate_gaussian(true_params, sample_size, intercept=FALSE)
maxIter <- sample_size * 2000
tol <- 1e-8
#################################
## SAG with Constant Step Size ##
#################################
test_that("constant step size SAG gradient norm is zero", {
    sag_fit <- sag_constant(sim$X, sim$y, lambda=lambda,
                            maxiter=maxIter, tol=tol, family=0)
    expect_less_than(norm(sag_fit$d, type="F"), eps)
})
#########################
## SAG with linesearch ##
#########################
test_that("linesearch SAG gradient norm is zero", {
    sag_fit <- sag_ls(sim$X, sim$y, lambda=lambda,
                      maxiter=maxIter, tol=tol, family=0)
    expect_less_than(norm(sag_fit$d, type="F"), eps)
})
##########################################################
## SAG with line-search and adaptive Lipschitz Sampling ##
##########################################################
## test_that("linesearch adaptive SAG gradient norm is zero", {
##     sag_fit <- sag_adaptive_ls(sim$X, sim$y, lambda=0,
##                                maxiter = NROW(sim$x) * iter_coef, family=0)
##     expect_less_than(norm(sag_fit$d, type="F"), eps)
## })
