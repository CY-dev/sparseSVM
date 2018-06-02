\name{sparseSVM-package}
\alias{sparseSVM-package}
\docType{package}
\title{Solution Paths for Sparse High-dimensional Support Vector Machine with Lasso or Elastic-Net Regularization}
\description{Fast algorithm for fitting solution paths for sparse SVM regularized by lasso or elastic-net that generate sparse solutions.}
\details{
\tabular{ll}{
Package: \tab sparseSVM\cr
Type: \tab Package\cr
Version: \tab 1.1-6\cr
Date: \tab 2018-06-01\cr
License: \tab GPL-3\cr
}
Accepts \code{X,y} data for binary classification and
produces the solution path over a grid of values of the regularization parameter \code{lambda}. Also provides functions for plotting, prediction and parallelized cross-validation.
}

\references{Yi, C. and Huang, J. (2017) 
  \emph{Semismooth Newton Coordinate Descent Algorithm for 
  Elastic-Net Penalized Huber Loss Regression and Quantile Regression},
  \url{https://www.tandfonline.com/doi/abs/10.1080/10618600.2016.1256816?journalCode=ucgs20} \cr
  \emph{Journal of Computational and Graphical Statistics} \cr
}
\author{
Congrui Yi and Yaohui Zeng \cr
Maintainer: Congrui Yi <eric.ycr@gmail.com>
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
