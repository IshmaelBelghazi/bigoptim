##' @export
##' @useDynLib bigoptim, .registration=TRUE
test_sparse_dset <- function() {
  library(Matrix)
  data(rcv1_train)
  sparse_mat <- head(rcv1_train$X[, 1:6])
  sparse_mat[1, 4] <- 1
  .Call("C_sparse_test", sparse_mat)
}
