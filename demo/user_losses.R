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
maxiter <- n * 10  ## 10 passes throught the dataset
lambda <- 1/n 
tol <- 0
## -----------------------------------------------------------------------------
## SAG binomial with Constant step size
print("Running Stochastic Average Gradient with constant step size")
sag_loss_time <- system.time({sag_constant_fit <- sag_fit(X=X, y=y, lambda=lambda, maxiter=maxiter,
                                                          tol=tol, family="binomial",
                                                          fit_alg="constant", standardize=FALSE)
})
cost_constant <- get_cost(sag_constant_fit, X, y)
print(sprintf("Cost is: %f.", cost_constant))
print(sag_loss_time)
## -----------------------------------------------------------------------------
## SAG with user supplied R functions and constant step size
binomial_loss <- function(y, innerProd) log(1 + exp(-y * innerProd))
binomial_grad <- function(y, innerProd) -y/(1 + exp(y * innerProd))

user_R_binomial <- make_R_loss(loss=binomial_loss,
                               grad=binomial_grad)
R_loss_time <- system.time({sag_constant_fit <- sag_fit(X=X, y=y, lambda=lambda, maxiter=maxiter,
                                                          tol=tol, family="R", fit_alg="constant",
                                                          standardize=FALSE,
                                                          user_loss_function=user_R_binomial)
})
cost_constant <- .get_cost(X, y, sag_constant_fit$w, lambda, "binomial",
                           backend="C")
print(sprintf("Cost is: %f", cost_constant))
print(R_loss_time)
## -----------------------------------------------------------------------------
## SAG with user supplied C functions and constant step size
## Inlining user supplid binomial C loss and loss gradient functions
src <- "
double user_binomial_loss(double y, double innerProd) {
return log(1 + exp(-y * innerProd));
}

double user_binomial_loss_grad(double y, double innerProd) {
return -y/(1 + exp(y * innerProd));
}
"
user_c_binomial <- make_c_loss(src,
                               loss_name="user_binomial_loss",
                               grad_name="user_binomial_loss_grad")
C_loss_time <- system.time({sag_constant_fit <- sag_fit(X=X, y=y, lambda=lambda, maxiter=maxiter,
                                                        tol=tol, family="c_shared",
                                                        user_loss_function=user_c_binomial,
                                                        fit_alg="constant", standardize=FALSE)
})
cost_constant <- .get_cost(X, y, sag_constant_fit$w, lambda, "binomial",
                           backend="C")
print(sprintf("Cost is: %f", cost_constant))
print(C_loss_time)
