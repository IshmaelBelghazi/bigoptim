## * Various utility functions
## ** Log-Sum- exp
.logsumexp_R <- function(x, ...) {
    max_x <- max(x, ...)
    max_x + log(sum(exp(x - max_x)))
}
## ** Sparse matrices check
##' @export
is.sparse <- function(X) inherits(X, "sparseMatrix")
