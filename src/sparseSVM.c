#include <math.h>
#include <time.h>
#include <string.h>
#include <R.h>
#include <R_ext/Applic.h>
#include "Rinternals.h"
#include "R_ext/Rdynload.h"

static double sign(double x) {
  if (x>0) return 1.0;
  else if (x<0) return -1.0;
  else return 0.0;
}

static double crossprod(double *yx, double *v, int n, int j) {
  int jn = j*n;
  double sum=0.0;
  for (int i=0;i<n;i++) sum += yx[jn+i]*v[i];
  return(sum);
}

// standardization of features
static void standardize(double *x, double *y, double *x2, double *yx, 
  double *sx_pos, double *sx_neg, double *syx, 
  double *shift, double *scale, int *nonconst, int n, int p) 
{
  int i, j, jn; 
  double xm, xsd, xvar, csum_pos, csum_neg, csum;
  for (j=1; j<p; j++) {
    jn = j*n; xm = 0.0; xvar = 0.0; 
    csum_pos = 0.0; csum_neg = 0.0; csum = 0.0;
    for (i=0; i<n; i++) xm += x[jn+i];
    xm /= n;
    for (i=0; i<n; i++) {
      x[jn+i] -= xm;
      x2[jn+i] = pow(x[jn+i], 2);
      xvar += x2[jn+i];
    }
    xvar /= n;
    xsd = sqrt(xvar);
    if (xsd > 1e-6) {
      nonconst[j] = 1;
      for (i=0; i<n; i++) {
        x[jn+i] = x[jn+i]/xsd;
        x2[jn+i] = x2[jn+i]/xvar;
        yx[jn+i] = y[i]*x[jn+i];
        if (y[i] > 0) {
          csum_pos += x[jn+i];
        } else {
          csum_neg += x[jn+i];
        }
        csum += yx[jn+i];
      }
      shift[j] = xm;
      scale[j] = xsd;
      sx_pos[j] = csum_pos;
      sx_neg[j] = csum_neg;
      syx[j] = csum;      
    }
  }
}

// rescaling of features
static void rescale(double *x, double *y, double *x2, double *yx, 
  double *sx_pos, double *sx_neg, double *syx, 
  double *shift, double *scale, int *nonconst, int n, int p) 
{
  int i, j, jn; 
  double cmin, cmax, crange, csum_pos, csum_neg, csum;
  for (j=1; j<p; j++) {
    jn = j*n;
    cmin = x[jn]; cmax = x[jn];
    csum_pos = 0.0; csum_neg = 0.0; csum = 0.0;
    for (i=1; i<n; i++) {
      if (x[jn+i] < cmin) {
        cmin = x[jn+i];
      } else if (x[jn+i] > cmax) {
        cmax = x[jn+i];
      }
    }
    crange = cmax - cmin;
    if (crange > 1e-6) {
      nonconst[j] = 1;
      for (i=0; i<n; i++) {
        x[jn+i] = (x[jn+i]-cmin)/crange;
        x2[jn+i] = pow(x[jn+i], 2);
        yx[jn+i] = y[i]*x[jn+i];
        if (y[i] > 0) {
          csum_pos += x[jn+i];
        } else {
          csum_neg += x[jn+i];
        }
        csum += yx[jn+i];
      }
      shift[j] = cmin;
      scale[j] = crange;
      sx_pos[j] = csum_pos;
      sx_neg[j] = csum_neg;
      syx[j] = csum;
    }
  }
}

// postprocessing of feature weights
static void postprocess(double *w, double *shift, double *scale, int *nonconst, int nlam, int p) {
  int l, j, lp; double prod;
  for (l = 0; l<nlam; l++) {
    lp = l*p;
    prod = 0.0;
    for (j=1; j<p; j++) {
      if (nonconst[j]) {
        w[lp+j] = w[lp+j]/scale[j];
        prod += shift[j]*w[lp+j];
      }
    }
    w[lp] -= prod;
  }
}

// Semismooth Newton Coordinate Descent (SNCD) for lasso/elastic-net regularized SVM
static void sparse_svm(double *w, int *iter, double *lambda, int *saturated, double *x, double *y, 
  double *pf, double *gamma_, double *alpha_, double *thresh_, double *lambda_min_, int *nlam_, int *n_, int *p_, 
  int *ppflag_, int *scrflag_, int *dfmax_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double gamma = gamma_[0]; double alpha = alpha_[0]; double thresh = thresh_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0]; int scrflag = scrflag_[0];
  int dfmax = dfmax_[0]; int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, lstart, lp, jn, num_pos, mismatch, nnzero = 0, violations = 0, nv = 0;
  double gi = 1.0/gamma, cmax, cmin, csum_pos, csum_neg, csum, pct, lstep, ldiff, lmax, l1, l2, v1, v2, v3, tmp, change, max_update, update, scrfactor = 1.0;  
  double *x2 = Calloc(n*p, double); // x^2
  double *sx_pos = Calloc(p, double); // column sum of x where y = 1
  double *sx_neg = Calloc(p, double); // column sum of x where y = -1
  double *yx = Calloc(n*p, double); // elementwise products: y[i] * x[i][j]
  double *syx = Calloc(p, double); // column sum of yx
  csum = 0.0; num_pos = 0;
  // intercept column
  for (i=0; i<n; i++) {
    x2[i] = 1.0;
    yx[i] = y[i];
    csum += yx[i];
    if (y[i] > 0) num_pos++;
  }
  syx[0] = csum;
  sx_pos[0] = num_pos;
  sx_neg[0] = n-num_pos;
  
  double *shift = Calloc(p, double);
  double *scale = Calloc(p, double);
  double *w_old = Calloc(p, double); 
  double *r = Calloc(n, double); // residual: 1-y(xw+b)
  double *s = Calloc(p, double);
  double *d1 = Calloc(n, double);
  double *d2 = Calloc(n, double);
  double *z = Calloc(p, double); // partial derivative used for screening: X^t*d1/n
  double cutoff;
  int *include = Calloc(p, int);
  int *nonconst = Calloc(p, int);

  // Preprocessing
  if (ppflag == 1) {
    standardize(x, y, x2, yx, sx_pos, sx_neg, syx, shift, scale, nonconst, n, p);
  } else if (ppflag == 2) {
    rescale(x, y, x2, yx, sx_pos, sx_neg, syx, shift, scale, nonconst, n, p);
  } else {
    for (j=1; j<p; j++) {
      jn = j*n; cmax = x[jn]; cmin = cmax; csum_pos = 0.0; csum_neg = 0.0; csum = 0.0;
      for (i=0; i<n; i++) {
        if (x[jn+i] > cmax) cmax = x[jn+i];
        else if (x[jn+i] < cmin) cmin = x[jn+i];
        x2[jn+i] = pow(x[jn+i], 2);
        yx[jn+i] = y[i]*x[jn+i];
        if (y[i] > 0) {
          csum_pos += x[jn+i];
        } else {
          csum_neg += x[jn+i];
        }
        csum += yx[jn+i];
      }
      sx_pos[j] = csum_pos;
      sx_neg[j] = csum_neg;
      syx[j] = csum;
      if (cmax - cmin > 1e-6) nonconst[j] = 1;
    }
  }
  
  //scrflag = 0: no screening; scrflag = 1: Adaptive Strong Rule(ASR); scrflag = 2: Strong Rule(SR)
  // ASR fits an appropriate scrfactor adaptively; SR always uses scrfactor = 1
  include[0] = 1; // always include an intercept
  if (scrflag == 0) {
    for (j=1; j<p; j++) if (nonconst[j]) include[j] = 1;
  } else {
    for (j=1; j<p; j++) if (!pf[j] && nonconst[j]) include[j] = 1;
  }
  
  // Initialization
  if (2*num_pos > n) {
    // initial intercept = 1
    w[0] = 1.0;
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
    w[0] = -1.0;
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
  if (user==0) {
    lmax = 0.0;
    if (2*num_pos > n) {
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
    if (*saturated) break;
    if (message) Rprintf("Lambda %d\n", l+1);
    lp = l*p;
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
        if (!include[j] && nonconst[j] && fabs(z[j]) > cutoff * pf[j]) include[j] = 1;
      }
      if (scrflag == 1) scrfactor = 0.0; //reset for ASR
    }
    while(iter[l] < max_iter) {
      // Check dfmax
      if (nnzero > dfmax) {
        for (int ll = l; ll<nlam; ll++) iter[ll] = NA_INTEGER;
        saturated[0] = 1;
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
              jn = j*n; v1 = 0.0; v2 = 0.0; pct = 0.0;
              for (i=0; i<n; i++) {
                v1 += yx[jn+i]*d1[i];
                v2 += x2[jn+i]*d2[i];
                pct += d2[i];
              }
      	      pct *= gamma/n; // percentage of residuals with absolute values below gamma
              if (pct < 0.05 || pct < 1.0/n) {
                // approximate v2 with a continuation technique
                for (i=0; i<n; i++) {
                  tmp = fabs(r[i]);
                  if (tmp > gamma) v2 += x2[jn+i]/tmp;
                }
              }
              v1 = (v1+syx[j])/(2.0*n); v2 /= 2.0*n;
              // Update w_j
              if (pf[j]==0.0) {
                // unpenalized
                w[lp+j] = w_old[j] + v1/v2;
              } else if (fabs(w_old[j]+s[j]) > 1.0) {
                s[j] = sign(w_old[j]+s[j]);
                w[lp+j] = w_old[j] + (v1-l1*pf[j]*s[j]-l2*pf[j]*w_old[j])/(v2+l2*pf[j]); 
              } else {
                s[j] = (v1+v2*w_old[j])/(l1*pf[j]);
                w[lp+j] = 0.0;
              }
              // mismatch between beta and s
              if (pf[j] > 0) {
                if (fabs(s[j]) > 1 || (w[lp+j] != 0 && s[j] != sign(w[lp+j]))) mismatch = 1;
              }
              // Update r, d1, d2 and compute candidate of max_update
              //Rprintf("l=%d, v1=%lf, v2=%lf, pct=%lf, change = %lf, mismatch = %d\n",l+1,v1,v2,pct,change,mismatch);
              change = w[lp+j]-w_old[j];
              if (change>1e-6) {
                for (i=0; i<n; i++) {
                  r[i] -= yx[jn+i]*change;
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
                w_old[j] = w[lp+j];
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
            v1 = (crossprod(yx, d1, n, j)+syx[j])/(2.0*n);
            // Check for KKT conditions
            if (fabs(v1)>l1*pf[j]) {
              include[j]=1;
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
          if (w[lp+j] != 0.0) nnzero++;
        }
        scrfactor /= alpha*ldiff;
        if (message) {
          if (violations) Rprintf("\n");
          Rprintf("Variable screening factor = %f\n", scrfactor);
        }
      } else {
        for (j=0; j<p; j++) if (w[lp+j] != 0.0) nnzero++;
      }
      if (message) Rprintf("# iterations = %d\n", iter[l]);
      if (violations==0) break;
      nv += violations;
    }
  }
  if (scrflag != 0 && message) Rprintf("# violations detected and fixed: %d\n", nv);
  // Postprocessing
  if (ppflag) postprocess(w, shift, scale, nonconst, nlam, p);
  
  Free(x2);
  Free(sx_pos);
  Free(sx_neg);
  Free(yx);
  Free(syx);
  Free(shift);
  Free(scale);
  Free(w_old);
  Free(r);
  Free(s);
  Free(d1);
  Free(d2);
  Free(z);
  Free(include);
  Free(nonconst);
}

static const R_CMethodDef cMethods[] = {
  {"sparse_svm", (DL_FUNC) &sparse_svm, 20},
  {NULL}
};

void R_init_sparseSVM(DllInfo *info)
{
  R_registerRoutines(info,cMethods,NULL,NULL,NULL);
}
