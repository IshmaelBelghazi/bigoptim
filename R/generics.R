######################
## Methods Generics ##
######################
## * Various getters
## ** Weights
## *** SAG_fit
##' @title Model's weights
##' Returns the model's weights. 
##' @param object object of class SAG
##' @param ... Any other pass-through parameters 
##' @export
coef.SAG_fit <- function(object, ...) object$w
## *** SAG
##' @title Model's weights
##' Returns the model's weights for each specified lambda.
##' @param object object of class SAG
##' @param ... Any other pass-through parameters 
##' @export
coef.SAG <- function(object, ...) {
  weights <- t(object$lambda_w)
  rownames(weights) <- paste("lambda=", object$input$lambdas)
  weights
}
##' ** Cost 
.get_cost <- function(X, y, w, lambda, family, backend) {
  switch(family,
         binomial={
           .binomial_cost(X=X, y=y, w=w,
                          lambda=lambda,
                          backend=backend)
         },
         gaussian={
           .gaussian_cost(X=X, y=y, w=w,
                          lambda=lambda,
                          backend=backend) 
         },
         exponential={
           .exponential_cost(X=X, y=y, w=w,
                          lambda=lambda,
                          backend=backend) 
         },
         poisson={
           .poisson_cost(X=X, y=y, w=w,
                         lambda=lambda,
                         backend=backend) 
         },
         stop("unrocognized family"))
}
##' @export
get_cost <- function(object, X, y, ...) {
  UseMethod('get_cost')
}
##' @export
get_cost.default <- function(object, X, y, ...) {
  stop("unkown object class")
}
##' @title Model's costs
##' Returns the model's true cost.
##' @param object object of class SAG_fit 
##' @param X Matrix of samples
##' @param y Matrix of targets
##' @param ... Any other pass-through parameters 
##' @export
get_cost.SAG_fit <- function(object, X, y, ...) {
  backend <- if(is.sparse(X)) "R" else "C"
  .get_cost(X, y,
            w=coef(object),
            lambda=object$input$lambda,
            family=object$input$family,
            backend=backend)
}
##' @title Model's cost
##' Returns the model's true cost for each specified lambda.
##' @param object object of class SAG
##' @param X Matrix of samples
##' @param y Matrix of targets
##' @param ... Any other pass-through parameters 
##' @export
get_cost.SAG <- function(object, X, y, ...) {
  backend <- if(is.sparse(X)) "R" else "C"
  lambdas <- object$input$lambdas
  iter <- 1
  costs <- apply(coef(object), 1, function(w) {
    cost <- .get_cost(X, y, w=w, lambda=lambdas[iter],
                      family=object$input$family,
                      backend=backend)
    iter <<- iter + 1
    cost
  })
  costs <- as.matrix(costs)
  colnames(costs) <- "cost"
  costs
}
##' ** Gradient
.get_grad <- function(X, y, w, lambda, family, backend) {
  switch(family,
         binomial={
           .binomial_cost_grad(X=X, y=y, w=w,
                               lambda=lambda,
                               backend=backend)
         }, 
         gaussian={
           .gaussian_cost_grad(X=X, y=y, w=w,
                               lambda=lambda,
                               backend=backend)
         },
         exponential={
           .exponential_cost_grad(X=X, y=y, w=w,
                                  lambda=lambda,
                                  backend=backend)
         },
         poisson={
           .poisson_cost_grad(X=X, y=y, w=w,
                              lambda=lambda,
                              backend=backend)
         },
         stop("unrecognized family"))
}
##' @export
get_grad <- function(object, X, y, ...) {
  UseMethod('get_grad')
}
##' @export
get_grad.default <- function(object, X, y, ...) {
  stop("unkown object class")
}
##' @title Model's gradient
##' Returns the model's true gradient.
##' @param object object of class SAG_fit 
##' @param X Matrix of samples
##' @param y Matrix of targets
##' @param ... Any other pass-through parameters 
##' @export
get_grad.SAG_fit <- function(object, X, y, ...) {
  backend <- if(is.sparse(X)) "R" else "C"
  .get_grad(X, y,
            w=coef(object),
            lambda=object$input$lambda,
            family=object$input$family,
            backend=backend)
}
##' @title Model's gradients
##' Returns gradients of model for each lambda.
##' @param object object of class SAG 
##' @param X Matrix of samples
##' @param y Matrix of targets
##' @param ... Any other pass-through parameters 
##' @export
get_grad.SAG <- function(object, X, y, ...) {
  backend <- if(is.sparse(X)) "R" else "C"
  lambdas <- object$input$lambdas
  iter <- 1
  grads <- t(apply(coef(object), 1, function(w) {
    grad <- .get_grad(X, y, w=w, lambda=lambdas[iter],
                      family=object$input$family,
                      backend=backend)
    iter <<- iter + 1
    grad
  }))
  grads
}
## ** Approximate Gradient
##' @export
get_approx_grad <- function(object, ...) {
  UseMethod('get_approx_grad')
}
##' @export
get_approx_grad.default <- function(object, ...) {
  stop("unrecognized object class")
}
##' @title get approximate gradient
##' Returns the models approximate gradient
##' @param  object of class SAG_fit
##' @param ... Any other pass-through parameters 
##' @export
get_approx_grad.SAG_fit <- function(object, ...) {
  object$d/NROW(object$g) + object$input$lambda * coef(object) 
}
