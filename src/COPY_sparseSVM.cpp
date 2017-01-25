#include "../inst/include/sparseSVM.h"
#include <math.h>
#include <time.h>
// #include <string.h>


static double sign(double x) {
  if (x>0) return 1.0;
  else if (x<0) return -1.0;
  else return 0.0;
}

// standardization of features
template <typename T>
static void standardize(SubMatrixAccessor<T> macc, 
                        const NumericVector &y,
                        NumericVector &sx_pos,
                        NumericVector &sx_neg, 
                        NumericVector &syx, 
                        NumericVector &shift, 
                        NumericVector &scale, 
                        LogicalVector &nonconst) {
  int n = macc.nrow();
  int p = macc.ncol();
  
  int i, j; 
  double tmp, xSum, xxSum, csum_pos, csum_neg;
  
  for (j = 0; j < p; j++) {                            
    xSum = xxSum = csum_pos = csum_neg = 0.0; 
    
    for (i = 0; i < n; i++) {
      tmp = macc(i, j);
      xxSum += tmp * tmp;
      if (y[i] > 0) { // y[i] == 1
        csum_pos += tmp;
      } else {        // y[i] == -1
        csum_neg += tmp;
      }
    }
    xSum = csum_pos + csum_neg;
    
    shift[j] = xSum / n;
    scale[j] = sqrt((xxSum - xSum * xSum / n) / (n-1));
    if (scale[j] > 1e-6) nonconst[j] = true;
    sx_pos[j] = csum_pos;
    sx_neg[j] = csum_neg;
    syx[j] = csum_pos - csum_neg;      
  }
  
  // unscaled intercept?
  shift[0] = 0;
  scale[0] = 1;
}


// postprocessing of feature weights
NumericMatrix& postprocess(NumericMatrix &w, 
                           const NumericVector &shift, 
                           const NumericVector &scale, 
                           const LogicalVector &nonconst,
                           int nlam, int p) {
  int l, j; 
  double prod;
  for (l = 0; l<nlam; l++) {
    prod = 0.0;
    for (j=1; j<p; j++) {
      if (nonconst[j]) {
        w(j, l) /= scale[j];
        prod += shift[j] * w(j, l);
      }
    }
    w(0, l) -= prod;
  }
  
  return w;
}

// Semismooth Newton Coordinate Descent (SNCD) for lasso/elastic-net regularized SVM
template <typename T>
List COPY_sparse_svm(SubMatrixAccessor<T> macc, NumericVector &lambda, const NumericVector &y, 
                     const NumericVector &pf, double gamma, double alpha, 
                     double thresh, double lambda_min,
                     int scrflag, int dfmax, int max_iter, bool user, bool message) {
  int n = macc.nrow();
  int p = macc.ncol();
  
  // returns
  int nlam = lambda.size();
  NumericMatrix w(p, nlam);
  IntegerVector iter(nlam);
  bool saturated = false;
  
  // Declarations
  int i, j, k, l, lstart, mismatch, nnzero = 0, violations = 0, nv = 0;
  double gi = 1.0/gamma, cmax, cmin, csum_pos, csum_neg, pct, lstep, ldiff, lmax, l1, l2, v1, v2, v3, tmp, change, max_update, update, scrfactor = 1.0;  
  NumericVector sx_pos(p); // column sum of x where y = 1
  NumericVector sx_neg(p); // column sum of x where y = -1
  NumericVector syx(p); // column sum of x*y
  
  NumericVector shift(p);
  NumericVector scale(p);
  NumericVector w_old(p);
  NumericVector r(n); // residual: 1-y(xw+b)
  NumericVector s(p);
  NumericVector d1(n);
  NumericVector d2(n);
  NumericVector z(p); // partial derivative used for screening: X^t*d1/n
  double cutoff;
  LogicalVector include(p);
  LogicalVector nonconst(p);
  
  // Preprocessing -> always standardize
  standardize(x, y, sx_pos, sx_neg, syx, shift, scale, nonconst);
  
  // scrflag = 0: no screening
  // scrflag = 1: Adaptive Strong Rule(ASR)
  // scrflag = 2: Strong Rule(SR)
  // ASR fits an appropriate scrfactor adaptively; SR always uses scrfactor = 1
  include[0] = true; // always include an intercept
  if (scrflag == 0) {
    for (j=1; j<p; j++) if (nonconst[j]) include[j] = true;
  } else {
    for (j=1; j<p; j++) if (!pf[j] && nonconst[j]) include[j] = true;
  }
  
  // Initialization
  if (2*sx_pos[0] > n) {
    // initial intercept = 1
    w(0, 0) = 1.0;
    w_old[0] = 1.0;
    for (i=0; i<n; i++) {
      if (y[i] > 0) {
        r[i] = 0.0;
        d1[i] = 0.0;
        d2[i] = gi;        
      } else {
        r[i] = 2.0;
        d1[i] = 1.0;
        d2[i] = 0.0;
      }
    }
  } else {
    // initial intercept = -1
    w(0, 0) = -1.0;
    w_old[0] = -1.0;
    for (i=0; i<n; i++) {
      if (y[i] > 0) {
        r[i] = 2.0;
        d1[i] = 1.0;
        d2[i] = 0.0;        
      } else {
        r[i] = 0.0;
        d1[i] = 0.0;
        d2[i] = gi;
      }
    }
  }
  
  // lambda
  if (!user) {
    lmax = 0.0;
    if (2*sx_pos[0] > n) {
      for (j=1; j<p; j++) {
        if (nonconst[j]) {
          z[j] = (2*sx_neg[j]-sx_pos[j])/(2*n);
          if (pf[j]) {
            tmp = fabs(z[j])/pf[j];
            if (tmp > lmax) lmax = tmp;
          }
        }
      }
    } else {
      for (j=1; j<p; j++) {
        if (nonconst[j]) {
          z[j] = (2*sx_pos[j]-sx_neg[j])/(2*n);
          if (pf[j]) {
            tmp = fabs(z[j])/pf[j];
            if (tmp > lmax) lmax = tmp;
          }
        }
      }
    }
    lmax /= alpha;
    lambda[0] = lmax;
    if (lambda_min == 0.0) lambda_min = 0.001;
    lstep = log(lambda_min)/(nlam - 1);
    for (l=1; l<nlam; l++) lambda[l] = lambda[l-1]*exp(lstep);
    lstart = 1;
  } else {
    lstart = 0;
  }
  
  // Solution path
  for (l=lstart; l<nlam; l++) {
    if (message) Rprintf("Lambda %d\n", l+1);
    l1 = lambda[l]*alpha;
    l2 = lambda[l]*(1.0-alpha);
    // Variable screening
    if (scrflag != 0) {
      if (scrfactor > 3.0) scrfactor = 3.0;
      if (l != 0) {
        cutoff = alpha*((1.0+scrfactor)*lambda[l] - scrfactor*lambda[l-1]);
        ldiff = lambda[l-1] - lambda[l];
      } else {
        cutoff = alpha*lambda[0];
        ldiff = 1.0;
      }
      for (j=1; j<p; j++) {
        if (!include[j] && nonconst[j] && fabs(z[j]) > cutoff * pf[j]) include[j] = true;
      }
      if (scrflag == 1) scrfactor = 0.0; //reset for ASR
    }
    while(iter[l] < max_iter) {
      // Check dfmax
      if (nnzero > dfmax) {
        for (int ll = l; ll<nlam; ll++) iter[ll] = NA_INTEGER;
        saturated = TRUE;
        break;
      }
      
      // Solve KKT equations on eligible predictors
      while(iter[l]<max_iter) {
        iter[l]++;
        mismatch = 0; max_update = 0.0;
        for (j=0; j<p; j++) {
          if (include[j]) {
            for (k=0; k<5; k++) {
              update = 0.0; mismatch = 0;
              // Calculate v1, v2
              v1 = 0.0; v2 = 0.0; pct = 0.0;
              for (i=0; i<n; i++) {
                tmp = macc(i, j);
                v1 += tmp * y[i] * d1[i];
                v2 += tmp * tmp * d2[i];
                pct += d2[i];
              }
              pct *= gamma/n; // percentage of residuals with absolute values below gamma
              if (pct < 0.05 || pct < 1.0/n) {
                // approximate v2 with a continuation technique
                for (i=0; i<n; i++) {
                  tmp = fabs(r[i]);
                  if (tmp > gamma) v2 += pow(macc(i, j), 2) / tmp;
                }
              }
              v1 = (v1+syx[j])/(2.0*n); v2 /= 2.0*n;
              // Update w_j
              if (pf[j]==0.0) {
                // unpenalized
                w(j, l) = w_old[j] + v1/v2;
              } else if (fabs(w_old[j]+s[j]) > 1.0) {
                s[j] = sign(w_old[j]+s[j]);
                w(j, l) = w_old[j] + (v1-l1*pf[j]*s[j]-l2*pf[j]*w_old[j])/(v2+l2*pf[j]); 
              } else {
                s[j] = (v1+v2*w_old[j])/(l1*pf[j]);
                w(j, l) = 0.0;
              }
              // mismatch between beta and s
              if (pf[j] > 0) {
                if (fabs(s[j]) > 1 || (w(j, l) != 0 && s[j] != sign(w(j, l)))) mismatch = 1;
              }
              // Update r, d1, d2 and compute candidate of max_update
              //Rprintf("l=%d, v1=%lf, v2=%lf, pct=%lf, change = %lf, mismatch = %d\n",l+1,v1,v2,pct,change,mismatch);
              change = w(j, l)-w_old[j];
              if (change>1e-6) {
                for (i=0; i<n; i++) {
                  r[i] -= macc(i, j) * y[i] * change;
                  if (fabs(r[i])>gamma) {
                    d1[i] = sign(r[i]);
                    d2[i] = 0.0;
                  } else {
                    d1[i] = r[i]*gi;
                    d2[i] = gi;
                  }
                }
                update = (v2+l2*pf[j])*change*change;
                if (update>max_update) max_update = update;
                w_old[j] = w(j, l);
              }
              if (!mismatch && update < thresh) break;
            }
          }
        }
        // Check convergence
        if (max_update < thresh) break;
      }
      // Scan for violations of the screening rule and count nonzero variables
      violations = 0; nnzero = 0;
      if (scrflag != 0) {
        for (j=0; j<p; j++) {
          if (!include[j] && nonconst[j]) {
            // crossprod
            v1 = 0; for (int i=0;i<n;i++) v1 += macc(i, j) * y[i] * d1[i];
            v1 = (v1 + syx[j]) / (2.0*n);
            // Check for KKT conditions
            if (fabs(v1)>l1*pf[j]) {
              include[j]=true;
              s[j] = v1/(l1*pf[j]);
              violations++;
              // pf[j] > 0
              // w_old = w = d = 0, no need for judgement
              if (message) Rprintf("+V%d", j);
            } else if (scrflag == 1) {
              v3 = fabs(v1-z[j]);
              if (v3 > scrfactor) scrfactor = v3;
            }
            z[j] = v1;
          }
          if (w_old[j] != 0.0) nnzero++;
        }
        scrfactor /= alpha*ldiff;
        if (message) {
          if (violations) Rprintf("\n");
          Rprintf("Variable screening factor = %f\n", scrfactor);
        }
      } else {
        for (j=0; j<p; j++) if (w_old[j] != 0.0) nnzero++;
      }
      if (message) Rprintf("# iterations = %d\n", iter[l]);
      if (violations==0) break;
      nv += violations;
    }
  }
  if (scrflag != 0 && message) Rprintf("# violations detected and fixed: %d\n", nv);
  // Postprocessing
  w = postprocess(w, shift, scale, nonconst, nlam, p);
  
  return List::create(w, iter, lambda, saturated);
}

// Dispatch function for COPY_cdfit_gaussian_hsr
// [[Rcpp::export]]
List COPY_cdfit_gaussian_hsr(XPtr<BigMatrix> xpMat, const NumericVector &y, 
                             const NumericVector &covar, NumericVector &lambda, 
                             const NumericVector &pf, double gamma, double alpha, 
                             double thresh, double lambda_min, int nlam, 
                             int scrflag, int dfmax, int max_iter, bool user, bool message) {
  switch(xpMat->matrix_type()) {
  case 1:
    return COPY_cdfit_gaussian_hsr(SubMatrixAccessor<char>(*xpMat, row_idx, covar),   
                                   lambda, y, pf, gamma, alpha, thresh, lambda_min, 
                                   scrflag, dfmax, max_iter, user, message);
  case 2:
    return COPY_cdfit_gaussian_hsr(SubMatrixAccessor<short>(*xpMat, row_idx, covar),   
                                   lambda, y, pf, gamma, alpha, thresh, lambda_min, 
                                   scrflag, dfmax, max_iter, user, message);
  case 4:
    return COPY_cdfit_gaussian_hsr(SubMatrixAccessor<int>(*xpMat, row_idx, covar),   
                                   lambda, y, pf, gamma, alpha, thresh, lambda_min, 
                                   scrflag, dfmax, max_iter, user, message);
  case 6:
    return COPY_cdfit_gaussian_hsr(SubMatrixAccessor<float>(*xpMat, row_idx, covar),   
                                   lambda, y, pf, gamma, alpha, thresh, lambda_min, 
                                   scrflag, dfmax, max_iter, user, message);
  case 8:
    return COPY_cdfit_gaussian_hsr(SubMatrixAccessor<double>(*xpMat, row_idx, covar),   
                                   lambda, y, pf, gamma, alpha, thresh, lambda_min, 
                                   scrflag, dfmax, max_iter, user, message);
  default:
    throw Rcpp::exception("unknown type detected for big.matrix object!");
  }
}

/******************************************************************************/
 