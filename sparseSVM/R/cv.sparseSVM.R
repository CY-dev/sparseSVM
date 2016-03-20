cv.sparseSVM <- function(X, y, ..., ncores = 1, eval.metric = c("me"),
                         nfolds = 10, fold.id, seed, trace = FALSE) {
 eval.metric <- match.arg(eval.metric)
 fit <- sparseSVM(X, y, ...)
 cv.args <- list(...)
 cv.args$lambda <- fit$lambda
 n <- length(y)
 if (!missing(seed)) set.seed(seed)
 if(missing(fold.id)) {
   if ((min(table(y)) > nfolds)) {
     ind1 <- which(y==1)
     ind0 <- which(y==-1)
     n1 <- length(ind1)
     n0 <- length(ind0)
     cv.ind1 <- ceiling(sample(1:n1)/n1*nfolds)
     cv.ind0 <- ceiling(sample(1:n0)/n0*nfolds)
     fold.id <- numeric(n)
     fold.id[y==1] <- cv.ind1
     fold.id[y==-1] <- cv.ind0
   } else {
     fold.id <- ceiling(sample(1:n)/n*nfolds)
   }
 }
 
 parallel <- FALSE
 if (ncores > 1) {
   max.cores <- detectCores()
   if (ncores > max.cores) {
     stop("The number of cores specified (", ncores, ") is larger than the number of avaiable cores (", max.cores, ")!")
   }
   cluster <- makeCluster(ncores)
   if (!("cluster" %in% class(cluster))) stop("cluster is not of class 'cluster'; see ?makeCluster")
   parallel <- TRUE
   cat("Start parallel computing for cross-validation...")
   clusterExport(cluster, c("fold.id", "X", "y", "cv.args"), 
                 envir=environment())
   clusterCall(cluster, function() {
     require(sparseSVM)
#      dyn.load("src/sparseSVM.so")
#      source("R/sparseSVM.R")
#      source("R/predict.sparseSVM.R")
     })
   fold.results <- parLapply(cl=cluster, X=1:nfolds, fun=cvf, XX=X, y=y, 
                             fold.id=fold.id, cv.args=cv.args, trace = trace)
   stopCluster(cluster)
 }
 
 ME <- matrix(NA, nrow = n, ncol = length(cv.args$lambda))
 for (i in 1:nfolds) {
   if (parallel) {
     res <- fold.results[[i]]
   } else {
     if (trace) cat("Starting CV fold #",i,sep="","\n")
     res <- cvf(i, X, y, fold.id, cv.args)
   }
   ME[fold.id == i, 1:res$nl] <- res$me
 }

 ## Return
 me <- apply(ME, 2, mean)
 me.se <- apply(ME, 2, sd) / sqrt(n)
 ind <- !is.na(me)
 me <- me[ind]
 me.se <- me.se[ind]
 lambda <- fit$lambda[ind]
 
 if (identical(eval.metric, 'me')) {
   cve <- me
   cvse <- me.se
   min <- which.min(me)
   lambda.min <- lambda[min]
 } else {
   stop("Current version only support \"eval.metric == me\": Misclassification Error.")
 }
 
 ## TODO: get other metrics: F1 score, precision, recall, confusion matrix.
 val <- list(cve = cve, cvse = cvse, lambda = lambda, fit = fit, min = min, 
             lambda.min = lambda.min, eval.metric = eval.metric,
             fold.id = fold.id)
 structure(val, class = 'cv.sparseSVM')
}

cvf <- function(i, XX, y, fold.id, cv.args, trace) {
  cv.args$X <- XX[fold.id != i, , drop = FALSE]
  cv.args$y <- y[fold.id != i]

  fit.i <- do.call("sparseSVM", cv.args)
  X2 <- XX[fold.id == i, , drop = FALSE]
  y2 <- y[fold.id == i]
  
  yhat <- matrix(predict(fit.i, X2, type = 'class'), length(y2))
  me <- (yhat != y2)
  list(me = me, nl = length(fit.i$lambda))
}

# # Test cv.sparseSVM
# dyn.load("src/sparseSVM.so")
# source("R/sparseSVM.R")
# source("R/plot.sparseSVM.R")
# source("R/predict.sparseSVM.R")
# require(parallel)
# 
# X = matrix(rnorm(1000*100), 1000, 100)
# b = 3
# w = 5*rnorm(10)
# eps = rnorm(1000)
# y = sign(b + drop(X[,1:10] %*% w + eps))
# 
# fit = sparseSVM(X, y)
# coef(fit, 0.01)
# predict(fit, X[1:5,], lambda = c(0.02, 0.01))
# 
# cv.fit1 <- cv.sparseSVM(X, y, ncores = 2, seed = 1234)
# cv.fit2 <- cv.sparseSVM(X, y, seed = 1234)
# stopifnot(all.equal(cv.fit1, cv.fit2))
# 
# plot(cv.fit1)
# predict(cv.fit1, X)


