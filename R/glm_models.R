## * Generalized Linear Models R function 
.make_glm_fun <- function(linkfun, ...) {

    ## Negative log likelihood
    cost <- function(X, y, weights, ...) {
        
    }
    ## Gradient of negative log likelihood
    grad <- function(X, y, weights, ...) {

    }

    return(list(cost=cost, grad=grad))
}


.get_glm_condprob <- function(X, y, W,
                              activation=function(x) x,
                              log_prob=FALSE,
                              ...) {
    stop("not implemented yet")
}
## ** Cost functions (negative log-likelihood)
.get_glm_cost <- function(X, y, W, activation=function(x) x, ...) {
    stop("not implemented yet")
}
## ** cost function gradient
.get_glm_grad <- function(X, y, W, activation=function(x) x, ...) {
    stop("not implemented yet")
}



