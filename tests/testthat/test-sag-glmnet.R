context("glmnet consistency tests")
library(glmnet)
## test parameters
eps <- 1e-02
## Simulating logistic datasets
true_params <- c(1, 2, 3)
sample_size <- 1000
sim <- .simulate_logistic(true_params, sample_size, intercept=FALSE)
lambda=0.1
## Fitting with glmnet
glm_fit <- glmnet(sim$X, as.factor(sim$y), family="binomial", intercept=FALSE,
                  lambda=0)
glmnet_hat <- as.matrix(coef(glm_fit))[-1, , drop=FALSE]
colnames(glmnet_hat) <- rownames(glmnet_hat) <- NULL
#################################
## SAG with Constant Step Size ##
#################################
test_that("constant sag and glmnet solutions are equal", {
    sag_fit <- sag_constant(sim$X, sim$y, lambda=0, maxiter=NROW(sim$X) * 200)
    expect_less_than(norm(glmnet_hat - sag_fit$w, type='F'), eps)
})
#########################
## SAG with linesearch ##
#########################
test_that("linesearch sag and glmnet solutions are equal", {
    sag_fit <- NULL
    expect_equal(sag_fit$w, glmnet_hat,
                 tolerance=eps, scale=1)

})
##########################################################
## SAG with line-search and adaptive Lipschitz Sampling ##
##########################################################
test_that("linesearch adaptive sag and glmnet solutions are equal", {
    sag_fit <- NULL
    expect_equal(sag_fit$w, glmnet_hat,
                 tolerance=eps, scale=1)

})
