\name{plot.sparseSVM}
\alias{plot.sparseSVM}
\title{Plot coefficients from a "sparseSVM" object}
\description{Produce a plot of the coefficient paths for a fitted
  \code{"sparseSVM"} object.}
\usage{
\method{plot}{sparseSVM}(x, xvar = c("lambda", "norm"), log.l = TRUE, nvars = TRUE, 
    alpha = 1, ...)
}
\arguments{
  \item{x}{A \code{sparseSVM} object.}
  \item{xvar}{What is on the X-axis. \code{"lambda"} plots against the lambda sequence, 
    \code{"norm"} against the L1-norm of the coefficients. Default is \code{"lambda"}.}
  \item{log.l}{Should \code{log(lambda)} be used instead of \code{lambda} when \code{xvar = "lambda"}?  Default is TRUE. It has no effect on \code{"norm"}.}
  \item{nvars}{If \code{TRUE} (the default), places an axis on top of the plot denoting the 
    number of variables with nonzero coefficients at each \code{lambda}.}
  \item{alpha}{A value between 0 and 1 for alpha transparency channel(0 means transparent 
    and 1 means opaque), helpful when the number of variables is large.}
  \item{\dots}{Other graphical parameters to \code{plot}.}
}
\author{Congrui Yi <congrui-yi@uiowa.edu>}
\seealso{\code{\link{sparseSVM}}}
\examples{
X = matrix(rnorm(1000*100), 1000, 100)
b = 3
w = 5*rnorm(10)
eps = rnorm(1000)
y = sign(b + drop(X[,1:10] \%*\% w + eps))

fit = sparseSVM(X, y)
par(mfrow = c(2,2))
plot(fit)
plot(fit, nvars = FALSE, alpha = 0.5)
plot(fit, log.l = FALSE)
plot(fit, xvar = "norm")
}
\keyword{models}
\keyword{classification}
\keyword{machine learning}
\keyword{SVM}
