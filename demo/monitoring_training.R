suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(glmnet))
## Loading Data set
data(covtype.libsvm)
## Normalizing Columns and adding intercept
X <- cbind(rep(1, NROW(covtype.libsvm$X)), scale(covtype.libsvm$X))
y <- covtype.libsvm$y
y[y == 2] <- -1
n <- NROW(X)
p <- NCOL(X)
## Setting seed
#set.seed(0)
## Setting up problem
n_passes <- 50  ## number of passses trough the dataset
maxiter <- n * n_passes
lambda <- 1/n 
tol <- 0
family <- "binomial"
fit_algs <- list(constant="constant",
                 linesearch="linesearch",
                 adaptive="adaptive")

sag_fits <- lapply(fit_algs, function(fit_alg) sag_fit(X, y,
                                                   lambda=lambda,
                                                   maxiter=maxiter,
                                                   family=family,
                                                   fit_alg=fit_alg,
                                                   standardize=FALSE,
                                                   tol=tol, monitor=TRUE))
print(lapply(sag_fits, function(sag_fit) get_cost(sag_fit, X, y)))
## Functions --------------------------------------------------------------------
make_monitor_table <- function(object, X, y, omit_init_state=TRUE) {
  monitor_fun <- list(cost=function(w)
                             .get_cost(X, y, w,
                                       object$input$lambda,
                                       family=family,
                                       backend="C"),
                      grad_norm=function(w)
                                  norm(.get_grad(X, y, w,
                                                 lambda=object$input$lambda,
                                                 family=family,
                                                 backend="C"),
                                       type='F'))
  monitor_w <- object$monitor_w
  if (omit_init_state) monitor_w <- monitor_w[, -1, drop=FALSE]
  
  mon <- lapply(monitor_fun, function(fun) apply(monitor_w, 2, fun))
  mon$effective_pass <- seq(0, length(mon$cost) - 1)
  if (omit_init_state) mon$effective_pass <- mon$effective_pass + 1
  
  mon$fit_alg <- rep(object$input$fit_alg, length(mon$cost))
  as.data.frame(mon)
}
## Glmnet training table
make_glmnet_monitor_table <- function(X, y, lambda, n_passes) {

  glmnet_fit <- glmnet(X, as.factor(y), alpha=0, family="binomial",
                       lambda=lambda, standardize=FALSE, intercept=FALSE)
  glmnet_w <- as.matrix(coef(glmnet_fit))[-1]
  glm_cost <- .get_cost(X, y, glmnet_w, lambda=lambda,
                        family=family, backend="C")
  glm_cost_grad <- norm(.get_grad(X, y, glmnet_w, lambda=lambda,
                                  family=family, backend="C"), 'F')
  effective_pass <- 1:n_passes
  data.frame(cost=rep(glm_cost, n_passes),
             grad_norm=rep(glm_cost_grad, n_passes),
             effective_pass=effective_pass,
             fit_alg="glmmnet")}


## Training monitoring functions
## Make graphs
make_graph_by_pass <- function(training_table) {

  training_table$fit_alg <- as.factor(training_table$fit_alg)
  ## Cost monitoring graph
  cost_graph <- ggplot(training_table,
                       aes(x=effective_pass,
                           y=cost,
                           color=fit_alg)) +
    geom_line() +
    xlab("Number of effective passes") +
    ylab("Cost") +
    ggtitle("Cost per effective pass through the covtype dataset") +
    scale_colour_discrete(name="Fit algorithm")

  ## Gradient Frobenius norm monitoring graph
  grad_norm_graph <- ggplot(training_table,
                       aes(x=effective_pass,
                           y=grad_norm,
                           color=fit_alg)) +
    geom_line() +
    scale_y_log10() +
    xlab("Number of effective passes") +
    ylab("gradient L2 norm (Log scale)") +
    ggtitle("Gradient L2 norm per effective pass through the covtype dataset") +
    scale_colour_discrete(name="Fit algorithm")


  list(cost=cost_graph, grad_norm=grad_norm_graph)
}
## Monitoring ------------------------------------------------------------------
training_tables <- lapply(sag_fits, function(sag_fit)
                                      make_monitor_table(sag_fit,
                                                         X, y, 
                                                         omit_init_state=TRUE))
## Adding glmnet estimation
training_tables$glmnet <- make_glmnet_monitor_table(X, y, lambda=lambda, n_passes=n_passes)
training_table <- do.call(function(...) rbind(... , make.row.names=FALSE), training_tables)
monitor_graphs <- make_graph_by_pass(training_table)
plot(monitor_graphs$cost)
plot(monitor_graphs$grad_norm)
