##' @title Stochastic Average Gradient Fit
##' @return object of class SAG_fit
##' @export
##' @useDynLib bigoptim C_sag_constant C_sag_linesearch
sag_fit <- function(X, y, lambda=0, maxiter=NULL, w=NULL, stepSize=NULL, stepSizeType=1,
                    iVals=NULL, d=NULL, g=NULL, covered=NULL, standardize=TRUE,
                    tol=1e-3, model="binomial", fit_alg="constant", ...) {

  if (length(grep("CMatrix", class(X))) > 0) {
    stop("sparse matrices support not implemented yet.")
  }

  ##,-------------------
  ##| Data preprocessing
  ##`-------------------
  if (standardize) {
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
    ##w <- matrix(rnorm(NCOL(X), mean=0, sd=1), nrow=NCOL(X), ncol=1)
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
             Lmax <- 0.25 * max(rowSums(X^2)) + lambda
             stepSize <- 1/Lmax
           }
           ## Calling C function
           sag_fit <- .Call("C_sag_constant", w, t(X), y, lambda, stepSize,
                            iVals, d, g, covered, as.integer(model_id), tol) 
         },
         linesearch={
           if (is.null(stepSize)) {
             stepSize <- 1  
           }
           ## Calling C function
           sag_fit <- .Call("C_sag_linesearch", w, t(X), y, lambda, stepSize, iVals, d, g, covered,
                            as.integer(stepSizeType), as.integer(model_id), tol)

         },
         adaptive={
           stop("not implemented yet")
         },
         stop("unrecognized fit algorithm"))
  
  ##,---------------------------
  ##| Structuring SAG_fit object
  ##`---------------------------
  sag_fit$params <- list(maxiter=maxiter, model=model, lambda=lambda, tol=tol, stepSize=stepSize)
  ## Adding approximate  cost gradient
  sag_fit$approx_grad <- sag_fit$d/NROW(X) + lambda * sag_fit$w
  class(sag_fit) <- "SAG_fit"
  sag_fit
}


## * SAG with constant Step Size
## logistic regression with SAG. constant step size
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
##' @useDynLib bigoptim C_sag_constant
sag_constant <- function(X, y, lambda=0,
                         maxiter=NULL, wInit=NULL,
                         stepSize=NULL, iVals=NULL,
                         d=NULL, g=NULL, covered=NULL,
                         family=1, tol=1e-3, ...) {

  if (length(grep("CMatrix", class(X))) > 0) {
    stop("sparse matrices support not implemented yet.")
  }

  if (is.null(maxiter)) {
    maxiter <- NROW(X) * 10
  }
  
  if (is.null(wInit)) {
    wInit <- matrix(0, nrow=NCOL(X), ncol=1)
    ##wInit <- matrix(rnorm(NCOL(X), mean=0, sd=1), nrow=NCOL(X), ncol=1)
  }

  if (is.null(stepSize)) {
    Lmax <- 0.25 * max(rowSums(X^2)) + lambda
    stepSize <- 1/Lmax
  }

  if (is.null(iVals)) {
    iVals <- matrix(sample.int(NROW(X), size=maxiter, replace=TRUE), nrow=maxiter, ncol=1)
  }

  if (is.null(d)) {
    d <- matrix(0, nrow=NCOL(X), ncol=1)
  }

  if (is.null(g)) {
    g <- matrix(0, nrow=NROW(X), ncol=1)
  }

  if (is.null(covered)) {
    covered <- matrix(0L, nrow=NROW(X), ncol=1)
  }

  ## Calling C function
  .Call("C_sag_constant", wInit, t(X), y, lambda, stepSize, iVals, d, g, covered,
        as.integer(family), tol)
}

## * Sag with Line-search
##' @export
##' @useDynLib  bigoptim C_sag_linesearch
sag_ls <- function(X, y, lambda=0, maxiter=NULL, wInit=NULL,
                   stepSize=NULL, iVals=NULL,
                   d=NULL, g=NULL, covered=NULL,
                   stepSizeType=1, family=1, tol=1e-3, ...) {
    
  if (length(grep("CMatrix", class(X))) > 0) {
    stop("sparse matrices support not implemented yet.")
  }
  if (is.null(maxiter)) {
    maxiter <- NROW(X) * 10
  }
  if (is.null(wInit)) {
    wInit <- matrix(0, nrow=NCOL(X), ncol=1)
  }
  if (is.null(stepSize)) {
    stepSize <- 1  
  }
  if (is.null(iVals)) {
    iVals <- matrix(sample.int(NROW(X), size=maxiter, replace=TRUE), nrow=maxiter, ncol=1)
    ## iVals <- ceiling(NROW(X) * matrix(runif(maxiter), ncol=1))
    ## iVals[] <- as.integer(iVals)
  }

  if (is.null(d)) {
    d <- matrix(0, nrow=NCOL(X), ncol=1)
  }
  if (is.null(g)) {
    g <- matrix(0, nrow=NROW(X), ncol=1)
  }
  if (is.null(covered)) {
    covered <- matrix(0L, nrow=NROW(X), ncol=1)
  }
  ## Calling C function
  .Call("C_sag_linesearch", wInit, t(X), y, lambda, stepSize, iVals, d, g, covered,
        as.integer(stepSizeType), as.integer(family), tol)
    
}
## * Sag with line-search and adaptive sampling
##' @export
##' @useDynLib  bigoptim C_sag_adaptive
sag_adaptive_ls <- function(X, y, lambda=0, Lmax=NULL,
                            Li=NULL, maxiter=NULL, randVals=NULL, wInit=NULL,
                            d=NULL, g=NULL, covered=NULL, increasing=TRUE, family=1, tol=1e-3, ...) {

    if (length(grep("CMatrix", class(X))) > 0) {
      stop("sparse matrices support not implemented yet.")
    }
    if (is.null(Li)) {
      ## Initial guess of overall Lipschitz Constant
      Lmax <- 1
    }
    if (is.null(Li)) {
      ## Initial guess of Lipschitz constant of each function
      Li <- matrix(1, nrow=NROW(X), ncol=1)
    }
    if (is.null(maxiter)) {
      maxiter <- NROW(X) * 10
    }
    if (is.null(randVals)) {
      randVals <- matrix(runif(maxiter * 2), nrow=maxiter, ncol=2)
    }
    if (is.null(wInit)) {
      wInit <- matrix(0, nrow=NCOL(X), ncol=1)
    }
    if (is.null(d)) {
      d <- matrix(0, nrow=NCOL(X), ncol=1)
    }
    if (is.null(g)) {
      g <- matrix(0, nrow=NROW(X), ncol=1)
    }
    if (is.null(covered)) {
      covered <- matrix(0L, nrow=NROW(X), ncol=1)
      ##covered[] <- as.integer(covered)
    }
    .Call("C_sag_adaptive", wInit, t(X), y, lambda, Lmax, Li,
          randVals, d, g, covered, increasing, tol)
}

## Error Checking
