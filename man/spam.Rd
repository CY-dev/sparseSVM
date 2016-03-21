\name{spam}
\alias{spam}
\title{Spam E-mail Classification}
\description{A data set collected at Hewlett-Packard Labs that classifies 4601
e-mails into spam or non-spam. It contains 57 other variables indicating the frequency of particular words and characters in each e-mail.

The first 48 variables are word frequencies in each e-mail of the form "freq_word".
The variables 49-54 indicate the frequency of characters `;', `(', `[', `!',
`$', and `#'. The variables 55-57 contain the average, longest 
and total run-length of consecutive capital letters. The last variable indicates the type of the
mail: either \code{"nonspam"} or \code{"spam"}.
}
\usage{data(spam)}
\format{A data frame with 4601 observations on 58 variables}

\details{
This data set contains 1813 e-mails classified as \code{"spam"}
and 2788 as \code{"nonspam"}. 
}
\source{
\url{http://statweb.stanford.edu/~tibs/ElemStatLearn/}
}
\references{
T. Hastie, R. Tibshirani, J.H. Friedman. \emph{The Elements of Statistical
Learning.} Springer, 2001.
}

\keyword{datasets}
