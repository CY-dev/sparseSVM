\name{predict.cv.sparseSVM}
\alias{predict.cv.sparseSVM}
\alias{coef.cv.sparseSVM}
\title{Model predictions based on "cv.sparseSVM" object.}
\description{This function returns fitted values, coefficients and more from a fitted \code{"cv.sparseSVM"} object.}
\usage{
\method{predict}{cv.sparseSVM}(object, X, lambda = object$lambda.min, 
        type = c("class","coefficients","nvars"), exact = FALSE, ...)
\method{coef}{cv.sparseSVM}(object, lambda = object$lambda.min, exact = FALSE, ...)
}
\arguments{
  \item{object}{Fitted \code{"cv.sparseSVM"} model object.}
  \item{X}{Matrix of values at which predictions are to be made. Used only for \code{type = "class"}.}
  \item{lambda}{Values of the regularization parameter \code{lambda} at which predictions 
    are requested. Default is the one corresponding to the minimum cross-validation error.}
  \item{type}{Type of prediction. \code{"class"} returns the class labels; 
    \code{"coefficients"} returns the coefficients; \code{"nvars"} returns the number of 
    nonzero coefficients at each value of \code{lambda}.}
  \item{exact}{If \code{exact=FALSE} (default), then the function uses linear interpolation 
    to make predictions for values of \code{lambda} that do not coincide with those used to 
    fit the model. If \code{exact=TRUE}, and predictions are requested at values of \code{lambda} 
    not included in the original fit, the model is refit on a lambda sequence consisting 
    \code{object$lambda} and the new ones before predictions are made. }
  \item{\dots}{Not used. Other arguments to predict.}
}
\value{The object returned depends on type.}
\author{
Congrui Yi and Yaohui Zeng \cr
Maintainer: Congrui Yi <eric.ycr@gmail.com>
}
\seealso{\code{\link{sparseSVM}}, \code{\link{cv.sparseSVM}}}
\examples{
X = matrix(rnorm(1000*100), 1000, 100)
b = 3
w = 5*rnorm(10)
eps = rnorm(1000)
y = sign(b + drop(X[,1:10] \%*\% w + eps))

cv.fit <- cv.sparseSVM(X, y, ncores = 2, seed = 1234)
predict(cv.fit, X)
predict(cv.fit, type = 'nvars')
predict(cv.fit, type = 'coef')
coef(cv.fit)

}
\keyword{models}
\keyword{classification}
\keyword{machine learning}
\keyword{SVM}
