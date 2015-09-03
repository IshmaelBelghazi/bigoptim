##' @export
make_R_loss <- function(loss, grad) {
  if (length(formals(loss)) !=  2)
    stop("loss function should have two variables")
  if (length(formals(grad)) != 2)
    stop("gradient function should have two variables")

  structure(list(R_loss_fun=loss,
                 R_loss_fun_env=environment(loss),
                 R_grad_fun=grad,
                 R_grad_fun_env=environment(grad)),
            class=c("user_R_loss"))
}
