\name{plot.cv.sparseSVM}
\alias{plot.cv.sparseSVM}
\title{Plot the cross-validation curve for a "cv.sparseSVM" object}
\description{Plot the cross-validation curve for a "cv.sparseSVM" object against the 
\code{lambda} values used, along with standard error bars.}
\usage{
\method{plot}{cv.sparseSVM}(x, log.x = TRUE, nvars = TRUE, ...)
}
\arguments{
  \item{x}{A \code{"cv.sparseSVM"} object.}
  \item{log.x}{Should X-axis be on the log scale?  Default is TRUE.}
  \item{nvars}{If \code{TRUE} (the default), places an axis on top of the plot denoting 
    the number of variables with nonzero coefficients at each \code{lambda}.}
  \item{\dots}{Other graphical parameters to \code{plot}}
}
\details{Produces a plot of mean cv errors at each \code{lambda} along with upper and lower standard error bars.}
\author{
Congrui Yi and Yaohui Zeng
}
\seealso{\code{\link{sparseSVM}}, \code{\link{cv.sparseSVM}}}
\examples{
X = matrix(rnorm(1000*100), 1000, 100)
b = 3
w = 5*rnorm(10)
eps = rnorm(1000)
y = sign(b + drop(X[,1:10] \%*\% w + eps))

cv.fit1 <- cv.sparseSVM(X, y, ncores = 2, seed = 1234)
plot(cv.fit1)
plot(cv.fit1, log.x = FALSE)
}
\keyword{models}
\keyword{classification}
