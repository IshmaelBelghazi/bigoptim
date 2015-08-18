##' @export
##' @useDynLib bigoptim, .registration=TRUE
test_sparse_dset <- function(dummy=TRUE) {
  library(Matrix)
  if (dummy) {
    sparse_mat <- Matrix(0, nrow=7, ncol=3, sparse=TRUE)
    sparse_mat[2, 1] <- 1
    sparse_mat[5, 1] <- 2
    sparse_mat[3, 2] <- 3
    sparse_mat[2, 3] <- 4
    sparse_mat[5, 3] <- 5
    sparse_mat[6, 3] <- 6
  } else {
    data(rcv1_train)
    sparse_mat <- rcv1_train$X
  }
    .Call("C_sparse_test", sparse_mat)
}
