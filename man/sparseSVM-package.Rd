\name{sparseSVM-package}
\alias{sparseSVM-package}
\docType{package}
\title{Solution Paths for Sparse Linear Support Vector Machine with Lasso or ELastic-Net Regularization}
\description{Fast algorithm for fitting solution paths for sparse linear SVM regularized by lasso or elastic-net that generate sparse solutions.}
\details{
\tabular{ll}{
Package: \tab sparseSVM\cr
Type: \tab Package\cr
Version: \tab 1.1-3\cr
Date: \tab 2016-05-13\cr
License: \tab GPL-3\cr
}
Accepts \code{X,y} data for binary classification and
produces the solution path over a grid of values of the regularization parameter \code{lambda}. Also provides functions for plotting, prediction and parallelized cross-validation.
}
\author{
Congrui Yi and Yaohui Zeng \cr
Maintainer: Congrui Yi <congrui-yi@uiowa.edu>
}
\examples{
X = matrix(rnorm(1000*100), 1000, 100)
b = 3
w = 5*rnorm(10)
eps = rnorm(1000)
y = sign(b + drop(X[,1:10] \%*\% w + eps))

fit = sparseSVM(X, y)
coef(fit, 0.05)
predict(fit, X[1:5,], lambda = c(0.2, 0.1))
plot(fit)

cv.fit <- cv.sparseSVM(X, y, ncores = 2, seed = 1234)
predict(cv.fit, X)
coef(cv.fit)
plot(cv.fit)
}
\keyword{models}
\keyword{classification}
\keyword{machine learning}
\keyword{SVM}
