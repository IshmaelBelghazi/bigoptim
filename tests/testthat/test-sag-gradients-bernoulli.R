context("Gradient tests -- BERNOULLI")
## test parameters
eps <- 1e-02
## data(covtype.libsvm)
## test_that("gradient of covtype.libsvm is close to 0", {
##   lambda <- 1
##   fit <- with(covtype.libsvm, sag_constant(X, y, lambda))
##   gradient <- with(covtype.libsvm, {
##     L2regularized.logistic.regression.gradient(X, y, lambda, fit$w)
##   })
##   expect_less_than(sum(abs(gradient)), eps)
## })

## data(rcv1_train)
## test_that("gradient of rcv1_train is close to 0", {
##   lambda <- 1
##   fit <- with(rcv1_train, sag_constant(X, y, lambda))
##   gradient <- with(rcv1_train, {
##     L2regularized.logistic.regression.gradient(X, y, lambda, fit$w)
##   })
##   expect_less_than(sum(abs(gradient)), eps)
## })

## Simulating logistic datasets
true_params <- c(1, 2, 3)
lambda <- 0
sample_size <- 1000
maxIter <- sample_size * 200
tol <- 1e-8
sim <- .simulate_bernoulli(true_params, sample_size, intercept=FALSE)

#################################
## SAG with Constant Step Size ##
#################################
test_that("constant step size SAG approximate gradient norm is zero", {
    sag_fit <- sag_constant(sim$X, sim$y, lambda=0,
                            maxiter=maxIter, tol=tol, family=1)
    expect_less_than(norm(sag_fit$d, type="F"), eps)
})
#########################
## SAG with linesearch ##
#########################
test_that("linesearch SAG approximate gradient norm is zero", {
    sag_fit <- sag_ls(sim$X, sim$y, lambda=0,
                      maxiter=maxIter, tol=tol, family=1)
    expect_less_than(norm(sag_fit$d, type="F"), eps)
})
##########################################################
## SAG with line-search and adaptive Lipschitz Sampling ##
##########################################################
## test_that("linesearch adaptive SAG gradient norm is zero", {
##     sag_fit <- sag_adaptive_ls(sim$X, sim$y, lambda=0,
##                                maxiter = NROW(sim$x) * iter_coef)
##     expect_less_than(norm(sag_fit$d, type="F"), eps)
## })
