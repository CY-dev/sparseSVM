\name{sparseSVM-package}
\alias{sparseSVM-package}
\docType{package}
\title{Solution Paths for Sparse Linear Support Vector Machine with Lasso or ELastic-Net Regularization}
\description{Fast algorithm for fitting solution paths for sparse linear SVM regularized by lasso or elastic-net that generate sparse solutions.}
\details{
\tabular{ll}{
Package: \tab sparseSVM\cr
Type: \tab Package\cr
Version: \tab 1.0-3\cr
Date: \tab 2016-03-19\cr
License: \tab GPL-3\cr
}
Very simple to use. Accepts \code{X,y} data for binary classification where \code{y} belongs to {+1, -1}, and
produces the regularized solution path over a grid of values for the regularization parameter \code{lambda}.
}
\author{Congrui Yi <congrui-yi@uiowa.edu>}
\examples{
X = matrix(rnorm(1000*100), 1000, 100)
b = 3
w = 5*rnorm(10)
eps = rnorm(1000)
y = sign(b + drop(X[,1:10] \%*\% w + eps))

fit = sparseSVM(X, y)
coef(fit, 0.01)
predict(fit, X[1:5,], lambda = c(0.02, 0.01))
}
\keyword{models}
\keyword{classification}
\keyword{machine learning}
\keyword{SVM}
