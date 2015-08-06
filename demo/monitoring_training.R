rm(list=ls())
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(reshape2))
suppressPackageStartupMessages(library(glmnet))
family <- 1  ## 1 for Bernoulli
## Loading Data set
data(covtype.libsvm)
## Normalizing Columns and adding intercept
X <- cbind(rep(1, NROW(covtype.libsvm$X)), scale(covtype.libsvm$X))
y <- covtype.libsvm$y
y[y == 2] <- -1
## Setting seed
set.seed(0)
## Setting up problem
## Number of lambdas to compute
nlambda <- 5
## Getting lambdas grid
glmnet_fit <- glmnet(X, as.factor(y), alpha=0, family="binomial",
                     nlambda=nlambda, standardize=FALSE, intercept=FALSE)
lambdas <- rev(glmnet_fit$lambda)

trainers <- list(constant=list(fun=sag_constant,
                               params=list(covered=NULL,
                                           w=NULL,
                                           d=NULL,
                                           g=NULL)),
                  ls=list(fun=sag_ls,
                          params=list(covered=NULL,
                                      w=NULL,
                                      d=NULL,
                                      g=NULL)))

## monitor_training <- function(X, y, lambda, iVals,
##                              trainers_list=trainers,
##                              maxiter=NROW(X) * 10,
##                              tol=0,
##                              training_periods=100,
##                              grad_fun=.bernoulli_cost_grad,
##                              cost_fun=.bernoulli_cost_C,
##                              verbose=TRUE, ...) {
##   iVals <- matrix(sample.int(NROW(X), size=maxiter, replace=TRUE),
##                   nrow=maxiter, ncol=1)
##   ## Number of training period (number of intervals from 1 to maxiter)
##   training_breaks <- c(seq(1, NROW(iVals) - NROW(iVals) %% training_periods,
##                            by=NROW(iVals) %/% training_periods), NROW(iVals))
##   training_table <- data.frame(iteration=numeric(),
##                               grad_norm=numeric(),
##                               cost=numeric(),
##                               fit_alg=character())
##   ## Training with warm starting.
##   for (i in 1:training_periods) {
##   if(verbose) {
##     print(sprintf("Processing training period: %d/%d", i, training_periods))
##   }
##     iVals_i <- iVals[training_breaks[i]:training_breaks[i + 1],, drop=FALSE]
##     ## Fitting model
##     fits_i <- lapply(trainers, function(model) {
##       fit <- model$fun(X, y, wInit=model$params$w, lambda=lambda,
##                        iVals=iVals_i,
##                        d=model$params$d,
##                        g=model$params$g,
##                        covered=model$params$covered,
##                        tol=tol,
##                        family=family)
##       fit <- list(params=fit[names(model$params)])
##       fit
##     })
##     trainers <- modifyList(trainers, fits_i, keep.null=TRUE)
##     ## Computing gradient norm and loss
##     results_i <- mapply(function(trainer, trainer_name) {
##       data.frame(iteration=training_breaks[i + 1],
##                  grad_norm=norm(grad_fun(X, y,
##                                          trainer$params$w,
##                                          lambda=lambda), 'F'),
##                  cost=cost_fun(X, y,
##                                trainer$params$w, lambda=lambda),
##                  fit_alg=trainer_name)},
##                  trainers, names(trainers),
##                  USE.NAMES=FALSE,
##                  SIMPLIFY=FALSE)
##     training_table <- rbind(training_table, do.call(rbind, results_i))
##   }
##   ## Adding glmnet
##   return(training_table)
## }


monitor_training <- function(X, y, lambda, iVals,
                             trainers_list=trainers,
                             maxiter=NROW(X) * 10,
                             tols,
                             grad_fun=.bernoulli_cost_grad,
                             cost_fun=.bernoulli_cost_C,
                             verbose=TRUE, ...) {
  iVals <- matrix(sample.int(NROW(X), size=maxiter, replace=TRUE),
                  nrow=maxiter, ncol=1)
  training_table <- data.frame(tol=numeric(),
                              grad_norm=numeric(),
                              cost=numeric(),
                              fit_alg=character())
  ## Training with warm starting.
  for (tol in tols) {
  if(verbose) {
    print(sprintf("Processing with tolerance: %f", tol))
  }
    ## Fitting model
    fits_i <- lapply(trainers, function(model) {
      fit <- model$fun(X, y, wInit=model$params$w, lambda=lambda,
                       iVals=iVals,
                       d=model$params$d,
                       g=model$params$g,
                       covered=model$params$covered,
                       tol=tol,
                       family=family)
      fit <- list(params=fit[names(model$params)])
      fit
    })
    trainers <- modifyList(trainers, fits_i, keep.null=TRUE)
    ## Computing gradient norm and loss
    results_i <- mapply(function(trainer, trainer_name) {
      data.frame(tol=tol,
                 grad_norm=norm(grad_fun(X, y,
                                         trainer$params$w,
                                         lambda=lambda), 'F'),
                 cost=cost_fun(X, y,
                               trainer$params$w, lambda=lambda),
                 fit_alg=trainer_name)},
                 trainers, names(trainers),
                 USE.NAMES=FALSE,
                 SIMPLIFY=FALSE)
    training_table <- rbind(training_table, do.call(rbind, results_i))
  }
  ## Adding glmnet
  return(training_table)
}
tols <- c(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
maxIter <- NROW(X) * 100
training_tables <- lapply(lambdas, function(lambda) monitor_training(X, y, lambda=lambda, iVals=iVals,
                                                                     maxiter=maxIter,
                                                                     trainers_list=trainers,
                                                                     grad_fun=.bernoulli_cost_grad_C,
                                                                     cost_fun=.bernoulli_cost_C,
                                                                     tols=tols))
names(training_tables) <- paste0("lambda_", lambdas)
training_tables <- lapply(training_tables, function(training_table) {
  training_table$fit_alg <- as.factor(training_table$fit_alg)
  training_table
})
## Make training monitoring graphs
make_training_graphs <- function(training_tables, log_grad_norm=FALSE) {
  training_graphs <- list()
  if (log_grad_norm) {
    iter_title <- "Approximate gradient log l2 norm"
    training_tables <- lapply(training_tables, function(table) {
      table$tol <- log(table$tol) 
    })
  } else {
    iter_title <- "Approximate gradient l2 norm"
  }
  for(table in names(training_tables)) {
    training_table <- training_tables[[table]]
    cost_graph <- ggplot(training_table, aes(x=tol,
                                             y=cost,
                                             color=fit_alg)) +
      geom_line() +
      xlab(iter_title) +
      scale_x_reverse() +
      ylab("Average l2 regularized Cost") +
      scale_y_log10() +
      ggtitle(paste("Cost vs", iter_title, table, sep=" "))
    grad_norm_graph <- ggplot(training_table, aes(x=tol,
                                                  y=grad_norm,
                                                  color=fit_alg)) +
      geom_line() +
      xlab(iter_title) +
      scale_x_reverse() +
      ylab("Frobenius norm of average l2 regularized cost gradient ")+
      scale_y_log10() + 
      ggtitle(paste0("Gradient norm vs", iter_title, table, sep=" "))
    ## Assigning graph to list
    training_graphs[[table]] <- list(cost=cost_graph,
                                     grad=grad_norm_graph)
  }
  return(training_graphs)
}

training_graphs <- make_training_graphs(training_tables, log_grad_norm=FALSE)
 
## Multiplot function
# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

## Plotting cost graphs
ncols <- ceiling(length(lambdas)/2)
multiplot(plotlist=lapply(training_graphs, function(graph) graph$cost), cols=ncols)
## Plotting Gradient graphs
multiplot(plotlist=lapply(training_graphs, function(graph) graph$grad), cols=ncols)
