\name{cv.sparseSVM}
\alias{cv.sparseSVM}
\title{
Cross validation for sparseSVM
}
\description{
Perform k-fold cross validation for sparse linear SVM regularized by lasso or elastic-net over a sequence of lambda values and find an optimal lambda.
}
\usage{
cv.sparseSVM(X, y, ..., ncores = 1, eval.metric = c("me"),
             nfolds = 10, fold.id, seed, trace = FALSE)
}
\arguments{
  \item{X}{Input matrix.}
  \item{y}{Response vector.}
  \item{...}{Additional arguments to \code{sparseSVM}.}
  \item{ncores}{\code{cv.sparseSVM} can be run in parallel across a
    cluster using the \code{parallel} package. If \code{ncores > 1},a cluster is 
    created to run \code{cv.sparseSVM} in parallel. The code is run in series if 
    \code{ncores = 1} (the default). An error occurs if \code{ncores} is larger than the
    total number of available cores.}
  \item{eval.metric}{The metric used to choose optimial \code{lambda}. Current version only
    supports "me": misclassification error.}
  \item{nfolds}{The number of cross-validation folds.  Default is 10.}
  \item{seed}{The seed of the random number generator in order to obtain 
  reproducible results.}
  \item{fold.id}{Which fold each observation belongs to.  By default the
    observations are randomly assigned by \code{cv.sparseSVM}.}
  \item{trace}{If set to TRUE, cv.sparseSVM will inform the user of its
    progress by announcing the beginning of each CV fold.  Default is
    FALSE. (No trace output when running in parallel even if \code{trace=TRUE}.)}
}
}
\details{
The function randomly partitions the data in nfolds. It calls hqreg nfolds+1 times, the first
to obtain the lambda sequence, and the remainder to fit with each of the folds left out once for
validation. The cross-validation error is the average of validation errors for the nfolds fits.

Note by default, the cross-validation fold assignments are balanced across the 1/-1 outcomes, so that each fold has the same proportion of 1/-1 outcomes (or as close to the same proportion as it is possible to achieve if cases do not divide evenly).
}
\value{
The function returns an object of S3 class "cv.sparseSVM", which is a list containing:
  \item{me}{The misclassification error for each value of lambda, averaged across the cross-validation folds}
  \item{lambda}{The values of lambda used in the cross-validation fits.}
  \item{fit}{The fitted \code{sparseSVM} object for the whole data.}
  \item{min}{The index of \code{lambda} corresponding to \code{lambda.min}.}
  \item{lambda.min}{The value of \code{lambda} with the minimum cross-validation error in terms of \code{eval.metric}.}
  \item{eval.metric}{The metric used in selecting optimal \code{lambda}.}
  \item{fold.id}{The same as above.}
}

\author{
Congrui Yi and Yaohui Zeng
}

\seealso{
\code{\link{sparseSVM}}
}
\examples{
X = matrix(rnorm(1000*100), 1000, 100)
b = 3
w = 5*rnorm(10)
eps = rnorm(1000)
y = sign(b + drop(X[,1:10] \%*\% w + eps))

cv.fit1 <- cv.sparseSVM(X, y, nfolds = 5, ncores = 4, seed = 1234)
cv.fit2 <- cv.sparseSVM(X, y, nfolds = 5, seed = 1234)
stopifnot(all.equal(cv.fit1, cv.fit2))

}
\keyword{models}
\keyword{classification}
\keyword{machine learning}
\keyword{SVM}