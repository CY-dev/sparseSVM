sparseSVM <- function (X, y, alpha = 1, gamma = 0.1, nlambda=100, lambda.min = ifelse(nrow(X)>ncol(X), 0.001, 0.01), lambda, 
                       preprocess = c("standardize", "rescale", "none"),  screen = c("ASR", "SR", "none"), max.iter = 1000, 
                       eps = 1e-5, dfmax = ncol(X)+1, penalty.factor=rep(1, ncol(X)), message = FALSE) {
  
  # Error checking
  preprocess <- match.arg(preprocess)
  screen <- match.arg(screen)
  if (alpha < 0 || alpha > 1) stop("alpha should be between 0 and 1")
  if (gamma < 0 || gamma > 1) stop("gamma should be between 0 and 1")
  if (missing(lambda) && nlambda < 2) stop("nlambda should be at least 2")
  if (length(penalty.factor)!=ncol(X)) stop("the length of penalty.factor should equal the number of columns of X")

  call <- match.call()
  # Include a column for intercept
  n <- nrow(X)
  XX <- cbind(rep(1,n), X)
  penalty.factor <- c(0, penalty.factor) # no penalty for intercept term
  p <- ncol(XX)
  
  if(missing(lambda)) {
    lambda <- double(nlambda)
    user <- 0
  } else {
    nlambda <- length(lambda)
    user <- 1
  }
  
  # Flag for preprocessing and screening
  ppflag = switch(preprocess, standardize = 1, rescale = 2, none = 0)
  scrflag = switch(screen, ASR = 1, SR = 2, none = 0)
  # Fitting
  fit <- .C("sparse_svm", double(p*nlambda), integer(nlambda), as.double(lambda), integer(1), integer(1), 
            as.double(XX), as.double(y), as.double(penalty.factor), as.double(gamma), as.double(alpha), 
            as.double(eps), as.double(lambda.min), as.integer(nlambda), as.integer(n), as.integer(p), 
            as.integer(ppflag), as.integer(scrflag), as.integer(dfmax), as.integer(max.iter), 
            as.integer(user), as.integer(message))
  weights <- matrix(fit[[1]],nrow = p)
  iter <- fit[[2]]
  lambda <- fit[[3]]
  saturated <- fit[[4]]
  nv <- fit[[5]]
  # Eliminate saturated lambda values
  ind <- !is.na(iter)
  weights <- weights[, ind]
  iter <- iter[ind]
  lambda <- lambda[ind]
  
  # Names
  vnames <- colnames(X)
  if (is.null(vnames)) vnames=paste0("V",seq(p-1))
  vnames <- c("(Intercept)", vnames)
  dimnames(weights) <- list(vnames, round(lambda, 4))

  # Output
  structure(list(call = call,
                 weights = weights,
                 iter = iter,
                 saturated = saturated,
                 lambda = lambda,
                 alpha = alpha,
                 gamma = gamma,
                 penalty.factor = penalty.factor[-1],
                 nv = nv),
            class = "sparseSVM")
}
