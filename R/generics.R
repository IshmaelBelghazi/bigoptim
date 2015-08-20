######################
## Methods Generics ##
######################
## * Various getters
## ** Weights
##' @export
coef.SAG_fit <- function(object, ...) object$w
##' ** Cost 
get_cost <- function(object, X, y, ...) {
  UseMethod('get_cost')
}
##' @export
get_cost.default <- function(object, X, y, ...) {
  stop("unkown object class")
}
##' @export
get_cost.SAG_fit <- function(object, X, y, ...) {
  backend <- if(is.sparse(object$X)) "R" else "C"
  switch(object$input$model,
         binomial={
           .binomial_cost(X=X, y=y, w=coef(object),
                          lambda=object$input$lambda,
                          backend=backend)
         },
         gaussian={
           .gaussian_cost(X=X, y=y, w=coef(object),
                          lambda=object$input$lambda,
                          backend=backend) 
         },
         exponential={
           stop("not implemented yet")
         },
         poisson={
           stop("not implemented yet")
         },
         stop("unrocognized model"))
}
##' ** Gradient
get_grad <- function(object, X, y, ...) {
  UseMethod('get_grad')
}
##' @export
get_grad.default <- function(object, X, y, ...) {
  stop("unkown object class")
}
##' @export
get_grad.SAG_fit <- function(object, X, y, ...) {
  backend <- if(is.sparse(object$X)) "R" else "C"
  switch(object$input$model,
         binomial={
           .binomial_cost_grad(X=X, y=y, w=coef(object),
                               lambda=object$input$lambda,
                               backend=backend)
         },
         gaussian={
           .gaussian_cost_grad(X=X, y=y, w=coef(object),
                               lambda=object$input$lambda,
                               backend=backend)
         },
         exponential={
           stop("not implemented yet")
         },
         poisson={
           stop("not implemented yet")
         },
         stop("unrecognized model")
         )
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
##' @export 
get_approx_grad.SAG_fit <- function(object, ...) {
  object$d/NROW(object$g) + object$input$lambda * coef(object) 
}
