context("glmnet consistency tests -- BERNOULLI")
suppressMessages(library(glmnet))
## test parameters
eps <- 1e-02
## Simulating logistic datasets
true_params <- c(1, 2, 3)
sample_size <- 1000
maxIter <- sample_size * 200
tol <- 1e-8
sim <- .simulate_bernoulli(true_params, sample_size, intercept=FALSE)
lambda <- 0
## Fitting with glmnet
glm_fit <- glmnet(sim$X, as.factor(sim$y), family="binomial", intercept=FALSE,
                  lambda=lambda)
glmnet_hat <- as.matrix(coef(glm_fit))[-1, , drop=FALSE]
colnames(glmnet_hat) <- rownames(glmnet_hat) <- NULL
#################################
## SAG with Constant Step Size ##
#################################
test_that("constant sag and glmnet solutions are equal", {
    sag_fit <- sag_constant(sim$X, sim$y, lambda=lambda,
                            maxiter=maxIter, tol=tol)
    expect_less_than(norm(glmnet_hat - sag_fit$w, type='F'), eps)
})
#########################
## SAG with linesearch ##
#########################
test_that("linesearch sag and glmnet solutions are equal", {
    sag_fit <-  sag_ls(sim$X, sim$y, lambda=lambda,
                       maxiter=maxIter, tol=tol)
    expect_less_than(norm(glmnet_hat - sag_fit$w, type='F'), eps)
})
##########################################################
## SAG with line-search and adaptive Lipschitz Sampling ##
##########################################################
## test_that("linesearch adaptive sag and glmnet solutions are equal", {
##     sag_fit <- sag_adaptive_ls(sim$X, sim$y, lambda=lambda,
##                                maxiter=NROW(sim$X) * iter_coef)
##     expect_less_than(norm(glmnet_hat - sag_fit$w, type='F'), eps)
## })
