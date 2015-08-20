## * Loss functions
## ** Gaussian
## TODO(Ishmael): Add cost. gradient in math reform and reference
##' @export
.gaussian_cost <- function(X, y, w, lambda=0, backend="R") {
  switch(backend,
         R={
           .R_gaussian_cost(X=X, y=y, w=w, lambda=lambda)
         },
         C={
           .C_gaussian_cost(X=X, y=y, w=w, lambda=lambda)
         },
         stop("unrecognized backend"))
}
##' @export
.gaussian_cost_grad <- function(X, y, w, lambda=0, backend="R") {
  switch(backend,
         R={
           .R_gaussian_cost_grad(X=X, y=y, w=w, lambda=lambda)
         },
         C={
           .C_gaussian_cost_grad(X=X, y=y, w=w, lambda=lambda)
         },
         stop("unrecognized backend"))
}

##' @export
.R_gaussian_cost <- function(X, y, w, lambda=0) {
  innerProd <- X %*% w
  losses <- 0.5 * (y - innerProd)^2
  loss <- sum(losses)/NROW(X) + 0.5 * lambda * sum(w^2)
}

##' @export
.R_gaussian_cost_grad <- function(X, y, w, lambda=0) {
  grad <- matrix(0, nrow=NROW(w), ncol=1)
  for (i in 1:NROW(X)) {
    term = -(y[i] - X[i, ] %*% w)
    grad <- grad + term * X[i, ]
  }
  grad/NROW(X) + lambda * w
}

##' @export
##' @useDynLib bigoptim, .registration=TRUE
.C_gaussian_cost <- function(X, y, w, lambda=0) {
  .Call("C_gaussian_cost", t(X), y, w, lambda)
}

##' @export
##' @useDynLib bigoptim, .registration=TRUE
.C_gaussian_cost_grad <- function(X, y, w, lambda=0) {
  .Call("C_gaussian_cost_grad", t(X), y, w, lambda)
}

## *** Binomial
## ** Binomial Cost
##' @export
.binomial_cost <- function(X, y, w, lambda=0, backend="R") {
  switch(backend,
         R={
           .R_binomial_cost(X=X, y=y, w=w, lambda=lambda)
         },
         C={
           .C_binomial_cost(X=X, y=y, w=w, lambda=lambda)
         },
         stop("unrecognized backend"))
}
.binomial_cost_grad <- function(X, y, w, lambda=0, backend="R") {
  switch(backend,
         R={
           .R_binomial_cost_grad(X=X, y=y, w=w, lambda=lambda)
         },
         C={
           .C_binomial_cost_grad(X=X, y=y, w=w, lambda=lambda)
         },
         stop("unrecognized backend"))
}
## Reference Schmidt (2014) and Bishop (2006)
## P(C_1|x) = \frac{1}{exp(-w^{t}x)}
## D = {y_n, x_n}_{n=1}^{N} where y_{n} \in {-1, 1}
## P(C_1|x) = P(y| x) = \frac{1}{exp(y w^{t}x)}
## E = -LL + reg = \frac{\sum_n=1^{N} log(1 + exp(-y^{n}w^{t}x_{n}))}{N} + 0.5 \lambda ||W||_{2}^{2}
## \nabla E = \frac{\sum_n=1^{N} \frac{-y_{n}x_{n}}{(1 + exp(y^{n}w^{t}x_{n}))}}{N} + \lambda W
##' @export
.R_binomial_cost <- function(X, y, w, lambda=0) {
  innerProd <- X  %*% w
  losses <- log(1 + exp(-y * innerProd))
  cost <- sum(losses)/NROW(X) + 0.5 * lambda * sum(w^2) 
  cost
}
##' @export
.R_binomial_cost_grad <- function(X, y, w, lambda=0) {
  grad <- matrix(0, nrow=NROW(w), ncol=1)
  for (i in 1:NROW(X)) {
    term <- -y[i]/(1 + exp(y[i] * (X[i, ] %*% w)))
    grad <- grad + term * X[i, ]
  }
  grad/NROW(X) + lambda * w
}

##' @export
##' @useDynLib bigoptim, .registration=TRUE
.C_binomial_cost <- function(X, y, w, lambda=0) {
  .Call("C_binomial_cost", t(X), y, w, lambda)
}
##' @export
##' @useDynLib bigoptim, .registration=TRUE
.C_binomial_cost_grad <- function(X, y, w, lambda=0) {
  .Call("C_binomial_cost_grad", t(X), y, w, lambda)
}
