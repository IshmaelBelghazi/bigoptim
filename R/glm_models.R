## * Generalized Linear Models R function 
## ** Gradient functions
## *** Gaussian
##' @export
.gaussian_grad <- function(X, y, w, lambda=0) {
  (X %*% w - y) + lambda * sum(w) 
}
## *** Bernoulli
##' @export
.bernoulli_grad <- function(X, y, w, lambda=0) {
  (-y/(1 + exp(y * (X %*% w)))) + lambda * sum(w)
}

## *** Exponential
##' @export
.exponential_grad <- function(X, y, w, lambda=0) {
  (-y * exp(y * (X %*% w))) + lambda * sum(w)
}
## *** Poisson
##' @export
.poisson_grad <- function(X, y, w, lambda=0) {
  (exp(X %*% w) - y) + lambda * sum(w)
}
