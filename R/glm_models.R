## * Generalized Linear Models R function 
## ** Gradient functions
## *** Gaussian
##' @export
.gaussian_grad <- function(X, y, w) {
  ((X %*% w) - y)/NROW(X) 
}
## *** Bernoulli
##' @export
.bernoulli_grad <- function(X, y, w) {
  (-y/(1 + exp(y * (X %*% w))))/NROW(X)
}

## *** Exponential
##' @export
.exponential_grad <- function(X, y, w) {
  (-y * exp(y * (X %*% w)))/NROW(X)
}
## *** Poisson
##' @export
.poisson_grad <- function(X, y, w) {
  (exp(X %*% w) - y)/NROW(X)
}
