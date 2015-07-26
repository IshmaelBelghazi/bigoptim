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
## Reference Schmidt (2014) and Bishop (2006)
## P(C_1|x) = \frac{1}{exp(-w^{t}x)}
## D = {y_n, x_n}_{n=1}^{N} and y_{n} \in {-1, 1}
## P(C_1|x) = P(y| x) = \frac{1}{exp(y w^{t}x)}
## E = -LL = \frac{\sum_n=1^{N} log(1 + exp(-y^{n}w^{t}x_{n}))}{N} + 0.5 \lambda ||W||_{2}^{2}
## \nabla E = \frac{\sum_n=1^{N} \frac{-y_{n}x_{n}}{(1 + exp(y^{n}w^{t}x_{n}))}}{N} + \lambda W
 
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

