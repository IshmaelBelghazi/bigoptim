## * Generalized Linear Models R function 
## ** Gradient functions
## *** Gaussian
## TODO(Ishmael): Add cost. gradient in math reform and reference
##' @export
.gaussian_loss <- function(X, y, w, lambda=0) {
  innerProd <- X %*% w
  losses <- 0.5 * (innerProd - y)^2
  loss <- sum(losses)/NROW(X) + 0.5 * lambda * sum(w^2)
}
##' @export
.gaussian_grad <- function(X, y, w, lambda=0) {
  grads <- diag(c(X %*% w - y)) %*% X 
  matrix(colMeans(grads) + lambda * sum(w), ncol=1)
}
## *** Bernoulli
##' @export
.bernoulli_loss <- function(X, y, w, lambda=0) {
  innerProd <- X  %*% w
  losses <- log(1 + exp(-y * innerProd))
  loss <- sum(losses)/NROW(X) + 0.5 * lambda * sum(w^2)
  
}
##' @export
.bernoulli_grad <- function(X, y, w, lambda=0) {
  grad <- matrix(0, nrow=NROW(w), ncol=1)
  for (i in 1:NROW(X)) {
    term <- -y[i]/(1 + exp(y[i] * (X[i, ] %*% w)))
    grad <- grad + term * X[i, ]
  }
  grad/NROW(X) + lambda * w
}

