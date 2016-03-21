\name{plot.cv.sparseSVM}
\alias{plot.cv.sparseSVM}
\title{Plot the cross-validation curve for a "cv.sparseSVM" object}
\description{Plot the cross-validation curve for a "cv.sparseSVM" object against the 
\code{lambda} values used, along with standard error bars.}
\usage{
\method{plot}{cv.sparseSVM}(x, log.l = TRUE, nvars = TRUE, ...)
}
\arguments{
  \item{x}{A \code{"cv.sparseSVM"} object.}
  \item{log.l}{Should \code{log(lambda)} be used instead of \code{lambda} when \code{xvar = "lambda"}?  Default is TRUE.}
  \item{nvars}{If \code{TRUE} (the default), places an axis on top of the plot denoting 
    the number of variables with nonzero coefficients at each \code{lambda}.}
  \item{\dots}{Other graphical parameters to \code{plot}}
}
\details{Produces a plot of mean cv errors at each \code{lambda} along with upper and lower standard error bars.}
\author{
Congrui Yi and Yaohui Zeng \cr
Maintainer: Congrui Yi <congrui-yi@uiowa.edu>
}
\seealso{\code{\link{sparseSVM}}, \code{\link{cv.sparseSVM}}}
\examples{
X = matrix(rnorm(1000*100), 1000, 100)
b = 3
w = 5*rnorm(10)
eps = rnorm(1000)
y = sign(b + drop(X[,1:10] \%*\% w + eps))

cv.fit <- cv.sparseSVM(X, y, ncores = 2, seed = 1234)
plot(cv.fit)
plot(cv.fit, log.l = FALSE)
}
\keyword{models}
\keyword{classification}
\keyword{machine learning}
\keyword{SVM}
