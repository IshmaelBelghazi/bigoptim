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
sag_constant_fit <- sag_fit(X=X, y=y, lambda=lambda, maxiter=maxiter,
                            tol=tol, family="binomial",
                            fit_alg="constant", standardize=FALSE)
cost_constant <- get_cost(sag_constant_fit, X, y)
print(sprintf("Cost is: %f. ", cost_constant))
## -----------------------------------------------------------------------------
## SAG with user supplied C functions with Constant step size
user_loss_function <- list(shared_lib_path="/home/ishmael/Dropbox/gsoc/gsoc2015/bigoptim/misc/temp/shlib_binomial_loss.so",
                            loss_function_name="shared_loss",
                            grad_function_name="shared_loss_grad")
sag_constant_fit <- sag_fit(X=X, y=y, lambda=lambda, maxiter=maxiter,
                            tol=tol, family="c_shared",
                            user_loss_function=user_loss_function,
                            fit_alg="constant", standardize=FALSE)
cost_constant <- .get_cost(X, y, sag_constant_fit$w, lambda, "binomial", backend="C")
print(sprintf("Cost is: %f. ", cost_constant))
