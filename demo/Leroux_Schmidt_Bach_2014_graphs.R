rm(list=ls())
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(reshape2))
suppressPackageStartupMessages(library(glmnet))
## Loading Data set
data(covtype.libsvm)
## Normalizing Columns and adding intercept
X <- cbind(rep(1, NROW(covtype.libsvm$X)), scale(covtype.libsvm$X))
y <- covtype.libsvm$y
y[y == 2] <- -1
## Setting seed
## Setting up problem
## Number of lambdas to compute
lambda <- 1/NROW(X)
tol <- 0
maxiter <- NROW(X) * 50
model <- "binomial"

## Data Structures --------------------------------------------------------------
algs <- list(## constant=list(fit_alg="constant",
                           ## params=list(covered=NULL,
                           ##             w=NULL,
                           ##             d=NULL,
                           ##             g=NULL,
                           ##             stepSize=NULL
                           ##             )),
              linesearch=list(fit_alg="linesearch",
                              params=list(covered=NULL,
                                          w=NULL,
                                          d=NULL,
                                          g=NULL,
                                          stepSize=NULL
                                          ))
             )
## Functions --------------------------------------------------------------------
## Training monitoring functions
monitor_training_by_iter <- function(X, y, lambda, 
                                     algs,
                                     maxiter=NROW(X) * 50,
                                     tol=0,
                                     training_periods=50,
                                     model="binomial",
                                     verbose=TRUE, ...) {

  iVals <- matrix(sample.int(NROW(X), size=maxiter, replace=TRUE),
                  nrow=maxiter, ncol=1)
  ## Number of training period (number of intervals from 1 to maxiter)
  training_breaks <- c(seq(1, NROW(iVals) - NROW(iVals) %% training_periods,
                           by=NROW(iVals) %/% training_periods), NROW(iVals))
  training_table <- data.frame(iteration=numeric(),
                               grad_norm=numeric(),
                               cost=numeric(),
                               fit_alg=character())
  ## Training with warm starting.
  for (i in 1:training_periods) {
  if(verbose) {
    print(sprintf("Processing training period: %d/%d", i, training_periods))
  }
    iVals_i <- iVals[training_breaks[i]:training_breaks[i + 1],, drop=FALSE]
    ## Fitting model
    fits_i <- lapply(algs, function(alg) {
      if (verbose) {
        print(sprintf("Processing alg: %s", alg$fit_alg))
      }
      fit <- sag_fit(X, y, lambda,
                     w=algs$params$w,
                     iVals=iVals_i,
                     d=alg$params$d, g=alg$params$g,
                     maxiter=NROW(iVals_i),
                     covered=alg$params$covered,
                     stepSize=alg$params$stepSize,
                     tol=tol,
                     model=model,
                     fit_alg=alg$fit_alg)
      fit
    })
    for (alg_name in names(algs)) {
      algs[[alg_name]]$params$w <- fits_i[[alg_name]]$w
      algs[[alg_name]]$params$d <- fits_i[[alg_name]]$d
      algs[[alg_name]]$params$g <- fits_i[[alg_name]]$g
      algs[[alg_name]]$params$covered <- fits_i[[alg_name]]$covered
      algs[[alg_name]]$params$stepSize <- fits_i[[alg_name]]$stepSize
    }
    ## Computing gradient norm and loss
    results_i <- lapply(names(algs), function(alg_name) {
      data.frame(iteration=training_breaks[i + 1],
                 grad_norm=norm(get_grad(fits_i[[alg_name]], X, y), 'F'),
                 cost=log(get_cost(fits_i[[alg_name]], X, y)),
                 fit_alg=alg_name)
    })
    training_table <- rbind(training_table, do.call(rbind, results_i))
  }
  ## Adding glmnet
  return(training_table)
}
## Make graphs
make_graph_by_iter <- function(training_table) {

  training_table$fit_alg <- as.factor(training_table$fit_alg)

  cost_graph <- ggplot(training_table,
                       aes(x=iteration,
                           y=cost,
                           color=fit_alg)) +
    geom_line() +
    xlab("iteration") +
    ylab("cost") +
    ggtitle("cost vs iterations")

  cost_graph
}
## Monitoring ------------------------------------------------------------------
training_table <- monitor_training_by_iter(X, y, lambda,
                                           algs=algs, maxiter=maxiter,
                                           model=model) 
cost_graph <- make_graph_by_iter(training_table)
plot(cost_graph)
