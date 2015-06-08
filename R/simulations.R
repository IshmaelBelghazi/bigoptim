##' @export
.simulate_logistic <- function(true_params, sample_size, intercept=FALSE) {

    n_params <- length(true_params)
    X <- matrix(rnorm(n_params * sample_size), ncol=n_params)
    if (intercept) {
        X[, 1] <- 1
    }
    z <- X %*% true_params
    pr <- 1/(1 + exp(-z))
    y <- 2 * rbinom(sample_size, 1, pr) - 1
    y <- matrix(y, ncol=1)
    return(list(X=X, y=y, true_params=true_params))
}
