##' @title Stochastic Average Gradient
##' @param X Matrix, possibly sparse of features.
##' @param y Matrix of targets.
##' @param lambda Scalar. L2 regularization parameter.
##' @param maxiter Maximum number of iterations.
##' @param w Matrix of weights.
##' @param alpha constant step-size. Used only when fit_alg == "constant"
##' @param stepSizeType scalar default is 1 to use 1/L, set to 2 to use 2/(L + n*myu). Only used when fit_alg="linesearch"
##' @param Li Scalar or Matrix.Initial individual Lipschitz approximation. 
##' @param Lmax Initial global Lipschitz approximation.
##' @param increasing Boolean. TRUE allows for both increase and decrease of lipschitz coefficients. False allows only decrease.
##' @param d Initial approximation of cost function gradient.
##' @param g Initial approximation of individual losses gradient.
##' @param covered Matrix of covered samples.
##' @param standardize Boolean. Scales the data if True
##' @param tol Real. Miminal required approximate gradient norm before convergence.
##' @param family One of "binomial", "gaussian", "exponential" or "poisson"
##' @param fit_alg One of "constant", "linesearch" (default), or "adaptive"
##' @param monitor Boolean. If TRUE returns matrix of weights after each effective pass through the dataset.
##' @param ... Any other pass-through parameters.
##' @export
##' @return object of class SAG_fit
##' @useDynLib bigoptim, .registration=TRUE
sag_fit <- function(X, y, lambda=0, maxiter=NULL, w=NULL, alpha=NULL,
                    stepSizeType=1, Li=NULL, Lmax=NULL, increasing=TRUE,
                    d=NULL, g=NULL, covered=NULL, standardize=FALSE,
                    tol=1e-7, family="binomial", fit_alg="constant",
                    monitor=FALSE, ...) {

  ## Checking  for sparsity
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
  if (is.null(maxiter)) {
    if (monitor) stop("monitoring not allowed with unbounded maximum iterations")
    maxiter <- .Machine$integer.max
  }
  ## Initializing weights
  if (is.null(w)) {
    w <- matrix(0, nrow=NCOL(X), ncol=1)
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
  family_id <- switch(family,
                      gaussian=0,
                      binomial=1,
                      exponential=2,
                      poisson=3,
                      stop("unrecognized model"))
  ##,-------------------
  ##| Setting fit_alg id 
  ##`-------------------
  fit_alg_id <- switch(fit_alg,
                       constant=0,
                       linesearch=1,
                       adaptive=2,
                       stop("unrecognized model"))
  ##,------------------------
  ##| Fit algorithm selection
  ##`------------------------
  switch(fit_alg,
         constant={
           if (is.null(alpha)) {
             Lmax <- 0.25 * max(Matrix::rowSums(X^2)) + lambda
             alpha <-  1/Lmax ## 1/(16 * Lmax)
           }
         },
         linesearch={
           if (is.null(Lmax)) {
             ## TODO(Ishmael): Confusion between Lmax and stepSize
             Li <- 1
           }
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
         },
         stop("unrecognized fit algorithm"))

  sag_fit <- .Call("C_sag_fit", w, Matrix::t(X), y, lambda,
                   alpha, as.integer(stepSizeType), Li, Lmax,
                   as.integer(increasing),
                   d, g, covered, tol,
                   as.integer(maxiter),
                   as.integer(family_id),
                   as.integer(fit_alg_id),
                   as.integer(sparse),
                   as.integer(monitor))

  ##,---------------------------
  ##| Structuring SAG_fit object
  ##`---------------------------
  sag_fit$input <- list(maxiter=maxiter, family=family, lambda=lambda, tol=tol, alpha=alpha, fit_alg=fit_alg)
  class(sag_fit) <- "SAG_fit"
  sag_fit
}

##' @title Stochastic Average Gradient with warm-starting
##' @param X Matrix, possibly sparse of features.
##' @param y Matrix of targets.
##' @param lambdas Vector. Vector of L2 regularization parameters. 
##' @param maxiter Maximum number of iterations.
##' @param w Matrix of weights.
##' @param alpha constant step-size. Used only when fit_alg == "constant"
##' @param stepSizeType scalar default is 1 to use 1/L, set to 2 to use 2/(L + n*myu). Only used when fit_alg="linesearch"
##' @param Li Scalar or Matrix.Initial individual Lipschitz approximation. 
##' @param Lmax Initial global Lipschitz approximation.
##' @param increasing Boolean. TRUE allows increase of Lipschitz coeffecient. False allows only decrease.
##' @param d Initial approximation of cost function gradient.
##' @param g Initial approximation of individual losses gradient.
##' @param covered Matrix of covered samples.
##' @param standardize Boolean. Scales the data if True
##' @param tol Real. Miminal required approximate gradient norm before convergence.
##' @param family One of "binomial", "gaussian", "exponential" or "poisson"
##' @param fit_alg One of "constant", "linesearch" (default), or "adaptive". 
##' @param ... Any other pass-through parameters.
##' @export
##' @return object of class SAG
##' @useDynLib bigoptim, .registration=TRUE
sag <- function(X, y, lambdas, maxiter=NULL, w=NULL, alpha=NULL,
                stepSizeType=1, Li=NULL, Lmax=NULL, increasing=TRUE,
                d=NULL, g=NULL, covered=NULL, standardize=FALSE,
                tol=1e-7, family="binomial", fit_alg="constant",
                ...) {

  lambdas <- sort(lambdas, decreasing=TRUE)
  ## Checking  for sparsity
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
  if (is.null(maxiter)) {
    maxiter <- .Machine$integer.max
  }
  ## Initializing weights
  if (is.null(w)) {
    w <- matrix(0, nrow=NCOL(X), ncol=1)
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
  family_id <- switch(family,
                      gaussian=0,
                      binomial=1,
                      exponential=2,
                      poisson=3,
                      stop("unrecognized model"))
  ##,-------------------
  ##| Setting fit_alg id 
  ##`-------------------
  fit_alg_id <- switch(fit_alg,
                       constant=0,
                       linesearch=1,
                       adaptive=2,
                       stop("unrecognized model"))
  ##,------------------------
  ##| Fit algorithm selection
  ##`------------------------
  switch(fit_alg,
         constant={
           if (is.null(alpha)) {
             ## TODO(Ishmael): Lmax depends on lambda for warmstarting 
             Lmax <- 0.25 * max(Matrix::rowSums(X^2)) + lambdas[1]
             alpha <-  1/Lmax ## 1/(16 * Lmax)
           }
         },
         linesearch={
           if (is.null(Lmax)) {
             Li <- 1
           }
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
         },
         stop("unrecognized fit algorithm"))

  sag_fits <- .Call("C_sag", w, Matrix::t(X), y, lambdas,
                    alpha, as.integer(stepSizeType), Li, Lmax,
                    increasing, d, g, covered, tol,
                    as.integer(maxiter),
                    as.integer(family_id),
                    as.integer(fit_alg_id),
                    as.integer(sparse))

  ##,---------------------------
  ##| Structuring SAG_fit object
  ##`---------------------------
  sag_fits$input <- list(maxiter=maxiter,
                         family=family,
                         lambdas=lambdas,
                         tol=tol,
                         alpha=alpha,
                         stepSizeType=stepSizeType,
                         fit_alg=fit_alg)
  class(sag_fits) <- "SAG"
  sag_fits
}
