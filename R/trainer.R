##' @title Stochastic Average Gradient Fit
##' @return object of class SAG_fit
##' @export
##' @useDynLib bigoptim, .registration=TRUE
sag_fit <- function(X, y, lambda=0, maxiter=NULL, w=NULL, stepSize=NULL,
                    stepSizeType=1, Li=NULL, Lmax=NULL, increasing=TRUE,
                    iVals=NULL, d=NULL, g=NULL, covered=NULL, standardize=FALSE,
                    randVals=NULL,
                    tol=1e-3, model="binomial", fit_alg="constant",
                    monitor=FALSE, ...) {
  ## if (length(grep("CMatrix", class(X))) > 0) {
  ##   stop("sparse matrices support not implemented yet.")
  ## }

  sparse <- is.sparse(X)
  ##,-------------------
  ##| Data preprocessing
  ##`-------------------
  if (standardize && !sparse) {
    X <- scale(X)
  }
  ##,------------------------------
  ##| Initializing common variables
  ##`------------------------------
  ## Initializing maximum iterations
  if (is.null(maxiter)) {
    maxiter <- NROW(X) * 10
  }
  ## Initializing weights
  if (is.null(w)) {
    w <- matrix(0, nrow=NCOL(X), ncol=1)
  }
  ## initializing stochastic index array
  if (is.null(iVals)) {
    iVals <- matrix(sample.int(NROW(X), size=maxiter, replace=TRUE), nrow=maxiter, ncol=1)
  }
  ## Initializing loss derivatives
  if (is.null(d)) {
    d <- matrix(0, nrow=NCOL(X), ncol=1)
  }
  ## Initializing sum of loss derivatives 
  if (is.null(g)) {
    g <- matrix(0, nrow=NROW(X), ncol=1)
  }
  ## Iniitializing covered values tracker
  if (is.null(covered)) {
    covered <- matrix(0L, nrow=NROW(X), ncol=1)
  }

  ##,-----------------
  ##| Setting model id 
  ##`-----------------
  model_id <- switch(model,
                     gaussian=0,
                     binomial=1,
                     exponential=2,
                     poisson=3,
                     stop("unrecognized model"))

  ##,------------------------
  ##| Fit algorithm selection
  ##`------------------------
  switch(fit_alg,
         constant={
           if (is.null(stepSize)) {
             Lmax <- 0.25 * max(Matrix::rowSums(X^2)) + lambda
             stepSize <-  1/Lmax ## 1/(16 * Lmax)
           }
           ## Calling C function
           sag_fit <- .Call("C_sag_constant", w, Matrix::t(X), y, lambda, stepSize,
                            iVals, d, g, covered, as.integer(model_id), tol,
                            as.integer(sparse), as.integer(monitor))
         },
         linesearch={
           if (is.null(stepSize)) {
             ## TODO(Ishmael): Confusion between Lmax and stepSize
             stepSize <- 1
           }
           ## Calling C function
           sag_fit <- .Call("C_sag_linesearch", w, Matrix::t(X), y, lambda, stepSize, iVals, d, g, covered,
                            as.integer(stepSizeType), as.integer(model_id), tol,
                            as.integer(sparse), as.integer(monitor))
         },
         adaptive={        
           if (is.null(Lmax)) {
             ## Initial guess of overall Lipschitz Constant
             Lmax <- 1
           }
           if (is.null(Li)) {
             ## Initial guess of Lipschitz constant of each function
             Li <- matrix(1, nrow=NROW(X), ncol=1)
           }
           if (is.null(randVals)) {
             randVals <- matrix(runif(maxiter * 2), nrow=maxiter, ncol=2)
           }
           
           sag_fit <- .Call("C_sag_adaptive", w, Matrix::t(X), y, lambda,
                            Lmax, Li, randVals,
                            d, g, covered, increasing, as.integer(model_id), tol,
                            as.integer(sparse), as.integer(monitor))
         },
         stop("unrecognized fit algorithm"))
  
  ##,---------------------------
  ##| Structuring SAG_fit object
  ##`---------------------------
  sag_fit$.call <- sapply(match.call(expand.dots=TRUE)[-1], deparse) 
  sag_fit$input <- list(maxiter=maxiter, model=model, lambda=lambda, tol=tol, stepSize=stepSize, fit_alg=fit_alg)
  class(sag_fit) <- "SAG_fit"
  sag_fit
}
## TEMPORARY -------------------------------------------------------------------
##' @export
##' @useDynLib bigoptim, .registration=TRUE
sag_constant_mark <- function(w, X, y, lambda, stepSize, iVals, d, g, covered) {
  .Call("C_sag_constant_mark", w, t(X), y, lambda, stepSize, iVals, d, g, covered)
}

## ## * Sag with line-search and adaptive sampling
## ##' @export
## ##' @useDynLib  bigoptim C_sag_adaptive_mark
## sag_adaptive_ls <- function(X, y, lambda=0, Lmax=NULL,
##                             Li=NULL, maxiter=NULL, randVals=NULL, wInit=NULL,
##                             d=NULL, g=NULL, covered=NULL, increasing=TRUE, family=1, tol=1e-3, ...) {

##     if (length(grep("CMatrix", class(X))) > 0) {
##       stop("sparse matrices support not implemented yet.")
##     }
##     if (is.null(Li)) {
##       ## Initial guess of overall Lipschitz Constant
##       Lmax <- 1
##     }
##     if (is.null(Li)) {
##       ## Initial guess of Lipschitz constant of each function
##       Li <- matrix(1, nrow=NROW(X), ncol=1)
##     }
##     if (is.null(maxiter)) {
##       maxiter <- NROW(X) * 10
##     }
##     if (is.null(randVals)) {
##       randVals <- matrix(runif(maxiter * 2), nrow=maxiter, ncol=2)
##     }
##     if (is.null(wInit)) {
##       wInit <- matrix(0, nrow=NCOL(X), ncol=1)
##     }
##     if (is.null(d)) {
##       d <- matrix(0, nrow=NCOL(X), ncol=1)
##     }
##     if (is.null(g)) {
##       g <- matrix(0, nrow=NROW(X), ncol=1)
##     }
##     if (is.null(covered)) {
##       covered <- matrix(0L, nrow=NROW(X), ncol=1)
##       ##covered[] <- as.integer(covered)
##     }
##     .Call("C_sag_adaptive_mark", wInit, t(X), y, lambda, Lmax, Li,
##           randVals, d, g, covered, increasing, tol)
## }

## Error Checking
