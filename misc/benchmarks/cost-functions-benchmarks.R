library(devtools)
library(microbenchmark)
load_all()


## Cost functions Benchmarks ---------------------------------------------------
true_params <- rnorm(n=100)
lambda <- runif(1,min=0, max=10)
sample_size <- 10000
sims <- lapply(list(binomial=.simulate_binomial,
                    gaussian=.simulate_gaussian,
                    exponential=.simulate_exponential,
                    poisson=.simulate_poisson),
               function(f) f(true_params, sample_size))
## Benchmarking function ------------------------------------------------------
benchmark_funs <- function(data_list, functions_list) {
  Map(function(data, fun) {
    times <- list(R=microbenchmark(fun(data$X, data$y, data$true_params,
                                       lambda=lambda, backend="R"), times=100),
                  C=microbenchmark(fun(data$X, data$y, data$true_params,
                                       lambda=lambda, backend="C"), times=100))
  sapply(times, function(time) summary(time)$mean)
  }, data=data_list, fun=functions_list)
}
## Cost functions benchmarks --------------------------------------------------
cost_funs <- list(binomial=.binomial_cost,
                  gaussian=.gaussian_cost,
                  exponential=.exponential_cost,
                  poisson=.poisson_cost)
cost_benchmarks <- as.data.frame(benchmark_funs(sims, cost_funs))
print("cost functions benchmark")
print(cost_benchmarks)
## Grad functions Benchmarks ---------------------------------------------------
grad_funs <- list(binomial=.binomial_cost_grad,
                  gaussian=.gaussian_cost_grad,
                  exponential=.exponential_cost_grad,
                  poisson=.poisson_cost_grad)
grad_benchmarks <- as.data.frame(benchmark_funs(sims, grad_funs))
print("grad functions benchmark")
print(grad_benchmarks)
