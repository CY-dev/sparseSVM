\name{sparseSVM}
\alias{sparseSVM}
\title{Fit sparse linear SVM with lasso or elasti-net regularization}
\description{Fit solution paths for sparse linear SVM regularized by lasso or elastic-net over 
a grid of values for the regularization parameter lambda.}
\usage{
sparseSVM(X, y, alpha = 1, gamma = 0.01, nlambda=100, 
          lambda.min = ifelse(nrow(X)>ncol(X), 0.001, 0.01), 
          lambda, preprocess = c("standardize", "rescale", "none"),  
          screen = c("ASR", "SR", "none"), max.iter = 1000, eps = 1e-5, 
          dfmax = ncol(X)+1, penalty.factor=rep(1, ncol(X)), message = FALSE)
}
\arguments{
  \item{X}{Input matrix.}
  \item{y}{Response vector.}
  \item{alpha}{The elastic-net mixing parameter that controls the relative contribution 
    from the lasso and the ridge penalty. It must be a number between 0 and 1. \code{alpha=1} 
    is the lasso penalty and \code{alpha=0} the ridge penalty.}
  \item{gamma}{The tuning parameter for huberization smoothing of hinge loss. Default is 0.01.}
  \item{nlambda}{The number of lambda values.  Default is 100.}
  \item{lambda.min}{The smallest value for lambda, as a fraction of lambda.max, the data 
    derived entry value. 
    Default is 0.001 if the number of observations is larger than the number of variables 
    and 0.01 otherwise.}
  \item{lambda}{A user-specified sequence of lambda values. Typical usage is to leave 
    blank and have the program automatically compute a \code{lambda} sequence based on 
    \code{nlambda} and \code{lambda.min}. Specifying \code{lambda} overrides this. This 
    argument should be used with care and supplied with a decreasing sequence instead of 
    a single value. To get coefficients for a single \code{lambda}, use \code{coef} or 
    \code{predict} instead after fitting the solution path with \code{sparseSVM}.
    %or performing k-fold CV with \code{cv.sparseSVM}.
    }
  \item{preprocess}{Preprocessing technique to be applied to the input. Either 
    "standardize" (default), "rescale" or "none" (see \code{Details}). The coefficients 
    are always returned on the original scale.}
  \item{screen}{Screening rule to be applied at each \code{lambda} that discards variables 
    for speed. Either "ASR" (default), "SR" or "none". "SR" stands for the strong rule, 
    and "ASR" for the adaptive strong rule. Using "ASR" typically requires fewer iterations 
    to converge than "SR", but the computing time are generally close. Note that the option 
    "none" is used mainly for debugging, which may lead to much longer computing time.}
  \item{max.iter}{Maximum number of iterations. Default is 1000.}
  \item{eps}{Convergence threshold. The algorithms continue until the maximum change in the
    objective after any coefficient update is less than \code{eps} times the null deviance. 
    Default is \code{1E-7}.}
  \item{dfmax}{Upper bound for the number of nonzero coefficients. The algorithm exits and 
    returns a partial path if \code{dfmax} is reached. Useful for very large dimensions.}
  \item{penalty.factor}{A numeric vector of length equal to the number of variables. Each 
    component multiplies \code{lambda} to allow differential penalization. Can be 0 for 
    some variables, in which case the variable is always in the model without penalization. 
    Default is 1 for all variables.}
  \item{message}{If set to TRUE,  sparseSVM will inform the user of its progress. This argument 
    is kept for debugging. Default is FALSE.}
}

\details{
  The sequence of models indexed by the regularization parameter \code{lambda} is fitted
  using a semismooth Newton coordinate descent algorithm. The objective function is defined 
  to be \deqn{\frac{1}{n} \sum hingeLoss(y_i (x_i' w + b)) + \lambda\textrm{penalty}(w).}{\sum hingeLoss(y_i (x_i' w + b))/n + \lambda*penalty(w).}
  where 
  \deqn{hingeLoss(t) = max(0, 1-t)} and the intercept \code{b} is unpenalized. 

  The program supports different types of preprocessing techniques. They are applied to 
  each column of the input matrix \code{X}. Let x be a column of \code{X}. For 
  \code{preprocess = "standardize"}, the formula is 
  \deqn{x' = \frac{x-mean(x)}{sd(x)};}{x' = (x-mean(x))/sd(x);}
  for \code{preprocess = "rescale"}, 
  \deqn{x' = \frac{x-min(x)}{max(x)-min(x)}.}{x' = (x-min(x))/(max(x)-min(x)).}
  The models are fit with preprocessed input, then the coefficients are transformed back
  to the original scale via some algebra.
}

\value{
  The function returns an object of S3 class \code{"sparseSVM"}, which is a list containing:
  \item{call}{The call that produced this object.}
  \item{weights}{The fitted matrix of coefficients.  The number of rows is equal to the number 
    of coefficients, and the number of columns is equal to \code{nlambda}. An intercept is included.}
  \item{iter}{A vector of length \code{nlambda} containing the number of iterations until 
    convergence at each value of \code{lambda}.}
  \item{saturated}{A logical flag for whether the number of nonzero coefficients has reached \code{dfmax}.}
  \item{lambda}{The sequence of regularization parameter values in the path.}
  \item{alpha}{Same as above.}
  \item{gamma}{Same as above.}
  \item{penalty.factor}{Same as above.}
}
\author{
Congrui Yi and Yaohui Zeng \cr
Maintainer: Congrui Yi <congrui-yi@uiowa.edu>
}
\seealso{\code{\link{plot.sparseSVM}}, \code{\link{cv.sparseSVM}}}
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
