context("Glm cost-functions")
## Testing Parameters ----------------------------------------------------------
tol <- 1e-6
## Generating Data -------------------------------------------------------------
true_params <- rnorm(n=3)
lambda <- runif(1,min=0, max=10)
sample_size <- 3000
sims <- lapply(list(binomial=.simulate_binomial,
                    gaussian=.simulate_gaussian,
                    exponential=.simulate_exponential,
                    poisson=.simulate_poisson),
               function(f) f(true_params, sample_size))

## Cost functions consistency --------------------------------------------------
cost_funs <- list(binomial=.binomial_cost,
                  gaussian=.gaussian_cost,
                  exponential=.exponential_cost,
                  poisson=.poisson_cost)
costs <- Map(function(data, fun) {
  list(R=fun(data$X, data$y, data$true_params, lambda=lambda, backend="R"),
       C=fun(data$X, data$y, data$true_params, lambda=lambda, backend="C"))
}, data=sims, fun=cost_funs)

test_that("R and C cost functions are consistent", {
         expect_equal(costs$binomial$R, costs$binomial$C, tolerance=tol, scale=1)
         expect_equal(costs$gaussian$R, costs$gaussian$C, tolerance=tol, scale=1)
         expect_equal(costs$exponential$R, costs$exponential$C, tolerance=tol, scale=1)
         expect_equal(costs$poisson$R, costs$poisson$C, tolerance=tol, scale=1)
         })

## Cost functions consistency --------------------------------------------------
grad_funs <- list(binomial=.binomial_cost_grad,
                  gaussian=.gaussian_cost_grad,
                  exponential=.exponential_cost_grad,
                  poisson=.poisson_cost_grad)
grads <- Map(function(data, fun) {
  list(R=fun(data$X, data$y, data$true_params, lambda=lambda, backend="R"),
       C=fun(data$X, data$y, data$true_params, lambda=lambda, backend="C"))
}, data=sims, fun=grad_funs)

test_that("R and C grad functions are consistent",{
         expect_equal(grads$binomial$R, grads$binomial$C, tolerance=tol, scale=1)
         expect_equal(grads$gaussian$R, grads$gaussian$C, tolerance=tol, scale=1)
         expect_equal(grads$exponential$R, grads$exponential$C, tolerance=tol, scale=1)
         expect_equal(grads$poisson$R, grads$poisson$C, tolerance=tol, scale=1)
         })
