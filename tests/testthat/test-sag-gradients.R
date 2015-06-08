context("Gradient tests")

## test parameters
eps <- 1e-08
## Simulating logistic datasets
true_params <- c(1, 2, 3)
sample_size <- 1000
sim <- .simulate_logistic(true_params, sample_size, intercept=FALSE)

#################################
## SAG with Constant Step Size ##
#################################
test_that("constant step size Sag gradient norm is zero", {
    ## Fitting SAG
    sag_fit <- sag_constant(sim$X, sim$y, lambda=0, maxiter=NROW(sim$X) * 100)
    expect_less_than(norm(sag_fit$d, type="F"), eps)
})

#########################
## SAG with linesearch ##
#########################
test_that("linesearch SAG gradient norm is zero", {

    expect_less_than(sag_ls(), eps)
})
##########################################################
## SAG with line-search and adaptive Lipschitz Sampling ##
##########################################################
test_that("linesearch adaptive sag gradient norm is zero", {
    expect_less_than(sag_adaptive_ls(), eps)
})
