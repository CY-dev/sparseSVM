predict.cv.sparseSVM <- function(object, X, lambda = object$lambda.min, 
                                 type=c("class","coefficients","nvars"), 
                                 exact = FALSE, ...) {
  type = match.arg(type)
  predict.sparseSVM(object$fit, X = X, lambda = lambda, 
                    type = type, exact = exact, ...)
}

coef.cv.sparseSVM <- function(object, lambda = object$lambda.min, exact = FALSE, ...) {
  coef.sparseSVM(object$fit, lambda = lambda, exact = exact, ...)
}

## test
# dyn.load("src/sparseSVM.so")
# source("R/sparseSVM.R")
# source("R/plot.sparseSVM.R")
# source("R/predict.sparseSVM.R")
# source("R/cv.sparseSVM.R")
# require(parallel)
# 
# X = matrix(rnorm(1000*100), 1000, 100)
# b = 3
# w = 5*rnorm(10)
# eps = rnorm(1000)
# y = sign(b + drop(X[,1:10] %*% w + eps))
# 
# fit = sparseSVM(X, y)
# coef(fit, 0.01)
# predict(fit, X[1:5,], lambda = c(0.02, 0.01))
# 
# cv.fit1 <- cv.sparseSVM(X, y, ncores = 4, seed = 1234)
# predict(cv.fit1, X)
# predict(cv.fit1, type = 'nvars')
# predict(cv.fit1, type = 'coef')
# coef(cv.fit1)

