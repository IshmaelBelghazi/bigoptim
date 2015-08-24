## * GLM generic
## Cost and gradient structure of GLMs cost functions differ only in the form of the individual loss_fun functions.
## Reference Schmidt (2014) and Bishop (2006)
## P(C_1|x) = \frac{1}{exp(-w^{t}x)}
## D = {y_n, x_n}_{n=1}^{N} where y_{n} \in {-1, 1}
## P(C_1|x) = P(y| x) = \frac{1}{exp(y w^{t}x)}
## E = -LL + reg = \frac{\sum_n=1^{N} log(1 + exp(-y^{n}w^{t}x_{n}))}{N} + 0.5 \lambda ||W||_{2}^{2}
## \nabla E = \frac{\sum_n=1^{N} \frac{-y_{n}x_{n}}{(1 + exp(y^{n}w^{t}x_{n}))}}{N} + \lambda W
.R_glm_cost <- function(X, y, w, lambda, loss_fun) {
  innerProd <- X %*% w
  losses <- loss_fun(y, innerProd)
  cost <- sum(losses)/NROW(X) + 0.5 * lambda * sum(w^2)
  cost
}
.R_glm_cost_grad <- function(X, y, w, lambda, loss_grad_fun) {
  innerProd <- X %*% w
  innerprod_grads <- loss_grad_fun(y, innerProd)
  grads <- X * as.vector(innerprod_grads)
  grad <- Matrix::colMeans(grads) + lambda * w
  as.matrix(grad)
}
## * GLM individual loss function
## ** Gaussian
.R_gaussian_loss <- function(y, innerProd) 0.5 * (y - innerProd)^2
.R_gaussian_loss_grad <- function(y, innerProd) -(y - innerProd)
## ** Binomial
.R_binomial_loss <- function(y, innerProd) log(1 + exp(-y * innerProd))
.R_binomial_loss_grad <- function(y, innerProd) -y/(1 + exp(y * innerProd)) 
## ** Exponential 
.R_exponential_loss <- function(y, innerProd) exp(-y * innerProd) 
.R_exponential_loss_grad <- function(y, innerProd) -y * exp(-y * innerProd)
## ** Poisson
.R_poisson_loss <- function(y, innerProd) exp(innerProd) - y * innerProd 
.R_poisson_loss_grad <- function(y, innerProd) exp(innerProd) - y
## TODO(Ishmael): Add cost. gradient in math reform and reference
## * GLM cost functions
## ** Gaussian
.gaussian_cost <- function(X, y, w, lambda=0, backend="R") {
  switch(backend,
         R={
           .R_glm_cost(X, y, w, lambda, .R_gaussian_loss)
         },
         C={
           if (is.sparse(X)) stop("sparse matrices not support in C backend")
           .Call("C_gaussian_cost", t(X), y, w, lambda)
         },
         stop("unrecognized backend"))
}
.gaussian_cost_grad <- function(X, y, w, lambda=0, backend="R") {
  switch(backend,
         R={
           .R_glm_cost_grad(X=X, y=y, w=w, lambda=lambda,
                            .R_gaussian_loss_grad)
         },
         C={
           if (is.sparse(X)) stop("sparse matrices not support in C backend")
           .Call("C_gaussian_cost_grad", t(X), y, w, lambda)
         },
         stop("unrecognized backend"))
}
## ** Binomial
.binomial_cost <- function(X, y, w, lambda=0, backend="R") {
  switch(backend,
         R={
           .R_glm_cost(X, y, w, lambda,
                       .R_binomial_loss)
         },
         C={
           if (is.sparse(X)) stop("sparse matrices not support in C backend")
           .Call("C_binomial_cost", t(X), y, w, lambda)
         },
         stop("unrecognized backend"))
}
.binomial_cost_grad <- function(X, y, w, lambda=0, backend="R") {
  switch(backend,
         R={
           .R_glm_cost_grad(X=X, y=y, w=w, lambda=lambda,
                            .R_binomial_loss_grad)
         },
         C={
           if (is.sparse(X)) stop("sparse matrices not support in C backend")
           .Call("C_binomial_cost_grad", t(X), y, w, lambda)
         },
         stop("unrecognized backend"))
}
## ** Exponential
.exponential_cost <- function(X, y, w, lambda=0, backend="R") {
  switch(backend,
         R={
           .R_glm_cost(X, y, w, lambda, .R_exponential_loss)
         },
         C={
           if (is.sparse(X)) stop("sparse matrices not support in C backend")
           .Call("C_exponential_cost", t(X), y, w, lambda)

         },
         stop("unrecognized backend"))
}
.exponential_cost_grad <- function(X, y, w, lambda=0, backend="R") {
  switch(backend,
         R={
           .R_glm_cost_grad(X=X, y=y, w=w, lambda=lambda,
                            .R_exponential_loss_grad)
         },
         C={
           if (is.sparse(X)) stop("sparse matrices not support in C backend")
           .Call("C_exponential_cost_grad", t(X), y, w, lambda)
         },
         stop("unrecognized backend"))
}
## ** Poisson
.poisson_cost <- function(X, y, w, lambda=0, backend="R") {
  switch(backend,
         R={
           .R_glm_cost(X, y, w, lambda, .R_poisson_loss)
         },
         C={
           if (is.sparse(X)) stop("sparse matrices not support in C backend")
           .Call("C_poisson_cost", t(X), y, w, lambda)
         },
         stop("unrecognized backend"))
}
.poisson_cost_grad <- function(X, y, w, lambda=0, backend="R") {
  switch(backend,
         R={
           .R_glm_cost_grad(X=X, y=y, w=w, lambda=lambda,
                            .R_poisson_loss_grad)
         },
         C={
           if (is.sparse(X)) stop("sparse matrices not support in C backend")
           .Call("C_poisson_cost_grad", t(X), y, w, lambda)
         },
         stop("unrecognized backend"))
}
