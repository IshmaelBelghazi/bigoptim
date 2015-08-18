data(rcv1_train)
X <- rcv1_train$X
X <- cBind(rep(1, NROW(X), X), X)
y <- rcv1_train$y
n <- NROW(X)
p <- NCOL(X)
## Setting seed
set.seed(0)
maxIter <- n * 20 
lambda <- 1/n
tol <- 0
print("Running Stochastic average gradient with constant step size\n")
## -----------------------------------------------------------------------------
## SAG with Constant step size
sag_constant_fit <- sag_fit(X=X, y=y, lambda=lambda, maxiter=maxIter,
                            tol=0, fit_alg="constant", model="binomial")
cost_constant <- .binomial_cost(X, y, coef(sag_constant_fit), lambda=lambda, backend="R")
print(sprintf("Cost is: %f. Value in Mark's matlab code: 0.201831",
              cost_constant))
