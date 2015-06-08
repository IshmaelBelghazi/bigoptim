## * SAG with constant Step Size
## logisitic regression with SAG with constant step size
##' let n be the number of example. p the number of features
##' @title SAG with constant step size for logistic regression
##' @param w vector of weights m X 1
##' @param X matrix of examples n X m. Will be transposed when .Call
##' @param y matrix of targets n x 1
##' @param lambda scalar regularisation parameter
##' @param stepSize scalar constant step size
##' @param iVals Sequenc of examples to choose maxiter X 1
##' @param d initial approximation of average gradient p X 1 (should be the sum
##of previous gradients)
##' @param g previous derivativers of loss.
##' @param covered whether the example has been visited. n X 1
##' @param sequence of examples to choose
##' @return list of results
##' @author Mohamed Ishmael Diwan Belghazi
##' @export
##' @useDynLib bigoptim SAG_logistic
sag_constant <- function(X, y, lambda=0,
                         maxiter=NULL, wInit=NULL,
                         stepSize=NULL, iVals=NULL,
                         d=NULL, g=NULL, covered=NULL) {
    if (is.null(maxiter)) {
        maxiter <- NROW(X) * 20
    }
    if (is.null(wInit)) {
        wInit <- matrix(0, nrow=NCOL(X), ncol=1)
    }
    if (is.null(stepSize)) {
        Lmax <- 0.25 * max(rowSums(X^2)) + lambda
        stepSize <- 1/Lmax
    }
    if (is.null(iVals)) {
        iVals <- ceiling(NROW(X) * matrix(runif(maxiter), ncol=1))
        storage.mode(iVals) <- "integer"
    }
    if (is.null(d)) {
        d <- matrix(0, nrow=NCOL(X), ncol=1)
    }
    if (is.null(g)) {
        g <- matrix(0, nrow=NROW(X), ncol=1)
    }
    if (is.null(covered)) {
        covered <- matrix(0, nrow=NROW(X), ncol=1)
        storage.mode(covered) <- "integer"
    }
    ## Calling C function
        .Call("SAG_logistic", wInit, t(X), y, lambda, stepSize, iVals, d, g, covered)
}

## * Sag with Line-search
##' @export
sag_ls <- function() {
    error("not implemented yet")
}
##' @export
## * Sag with line-search and adaptive sampling
sag_adaptive_ls <- function() {
    error("not implemented yet")
}
