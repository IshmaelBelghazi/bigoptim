## .onLoad <- function(lib, pkg) {
##   library.dynam("bigoptim", pkg, lib)
##   setDefaults()
## }

onunload <- function (libpath) {
  library.dynam.unload("bigoptim", libpath)
}
