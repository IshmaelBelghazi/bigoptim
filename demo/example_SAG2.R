family <- 1  ## 1 for Bernoulli
## Loading Data set
data(covtype.libsvm)
## Normalizing Columns and adding intercept
X <- cbind(rep(1, NROW(covtype.libsvm$X)), scale(covtype.libsvm$X))
y <- covtype.libsvm$y
y[y == 2] <- -1
n <- NROW(X)
p <- NCOL(X)
## Setting seed
set.seed(0)
## Setting up problem
maxiter <- n * 10  ## 10 passes throught the dataset
lambda <- 1/n 
tol <- 0
## SAG with Constant step size
print("Running Stochastic Average Gradient with constant step size")
sag_constant_fit <- sag_constant(X=X, y=y, lambda=lambda, maxiter=maxiter,
                                 family=family, tol=tol)
w_constant <- sag_constant_fit$w
loss_constant <- .bernoulli_loss(X, y, w_constant, lambda=lambda)
print(sprintf("Loss is: %f. Value in Mark's matlab code: 0.513607",
              loss_constant))
## TEMPORARY -- START
print("Using C cost function")
C_cost_const <-  .bernoulli_cost_C(X, y, w_constant, lambda=lambda)
print(C_cost_const)
R_cost_grad_const <- .bernoulli_grad(X, y, w_constant, lambda=lambda)
R_cost_grad_const_norm <- norm(R_cost_grad_const, 'F')
print(sprintf("Gradient norm using R: %f. Value in Mark's matlab code: 0.001394", R_cost_grad_const_norm))
C_cost_grad_const <- .bernoulli_cost_grad_C(X, y, w_constant, lambda=lambda)
C_cost_grad_const_norm <- norm(C_cost_grad_const, 'F')
print(sprintf("Gradient norm using C: %f. Value in Mark's matlab code: 0.001394", C_cost_grad_const_norm))
## TEMPORARY -- END
## SAG with linesearch
print("Running Stochastic Average Gradient with line-search")
Lmax <- 1
sag_ls_fit <- sag_ls(X=X, y=y, lambda=lambda,
                     stepSize=Lmax, stepSizeType=1,
                     maxiter=maxiter, family=family,
                     tol=tol)
w_ls <- sag_ls_fit$w
loss_ls <- .bernoulli_loss(X, y, w_ls, lambda=lambda)
print(sprintf("Loss is: %f. Value in Mark's matlab code: 0.513497",
              loss_ls))
## SAG with linesearch and adaptive sampling
print(paste0("Running Stoachastic Average Gradient with ",
             "linesearch and adaptive sampling"))
print("Not implemented yet")



 
