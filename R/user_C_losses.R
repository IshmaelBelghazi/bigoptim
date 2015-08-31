##' @title Make user supplied C loss and gradient
##' @param src C source code as character string
##' @param loss_name Loss function name
##' @param grad_name Gradient function name
##' @export
make_c_loss <- function(src, loss_name, grad_name) {
  if (.Platform$OS.type !=  "unix")
    stop("user specified C loss function are currently only supported on POSIX systems")
  ## Determine R temporary directory and file
  code_dir <- tempdir()
  code_file <- tempfile(fileext=".c")
  if (file.exists(code_file)) {
    file.remove(code_file)
  }
  ## check for headers in supplied sources
  headers <- c("#include <R.h>",
               "#include <Rdefines.h>",
               "#include <Rmath.h>",
               "#include <R_ext/BLAS.h>")
  ## Adding missing headers
  src <- paste(c(headers[sapply(headers, function(header) !grepl(header, src))], src),
               collapse="\n")
  ## Write source to temporary file
  write(src, code_file)
  ## Reading file
  ## file.show(code_file)
  ## Setting compile command 
  compile_cmd <- paste0(R.home(component="bin"),
                        "/R CMD SHLIB ",
                        basename(code_file))
  ## Compiling file
  current_wd <- getwd(); on.exit(setwd(current_wd))
  setwd(code_dir)
  compile_output <- system(compile_cmd, intern=TRUE)
  lib_file <- gsub("\\.c", .Platform$dynlib.ext ,code_file)
  if (!file.exists(lib_file)) {
    cat("============= Compile info ==============")
    cat("\n")
    cat(compile_output)
    cat("\n")
    cat("=========================================")
    cat("\n")
    stop("compiled shared library file not found")
  }
  ## Checking for supplied symbol name
  if (!grepl(loss_name, src) || !grepl(grad_name, src)) {
    warning("provided loss function or gradient function name not found in source")
  }
  structure(list(src=src,
                 code_file_path=code_file,
                 lib_file_path=lib_file,
                 loss_name=loss_name,
                 grad_name=grad_name,
                 compile_output=compile_output),
            class=c("user_c_loss"))
}
