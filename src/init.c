  #include <math.h>
#include <time.h>
#include <string.h>
#include <R.h>
#include <R_ext/Applic.h>
#include "Rinternals.h"
#include "R_ext/Rdynload.h"

double sign(double x);
double crossprod(double *yx, double *v, int n, int j);

// Fit the initial solutions for unpenalized features in elastic-net penalized models
void init_svm(double *w, double *w_old, int *iter, double *yx, double *x2, double *y, 
    double *syx, double *r, double *pf, double *d1, double *d2, int *nonconst, 
    double gamma, double thresh, int n, int p, int num_pos, int max_iter)
{
  double gi = 1.0/gamma, v1, v2, pct, tmp, change, max_update, update; int i, j, k, jn;
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
  while (iter[0] < max_iter) {
    iter[0]++;
    max_update = 0.0;
    for (j=0; j<p; j++) {
      if (pf[j] == 0.0 && nonconst[j]) { // unpenalized
        for (k=0; k<5; k++) {
          update = 0.0;
          // Calculate v1, v2
          jn = j*n; v1 = 0.0; v2 = 0.0; pct = 0.0;
          for (i=0; i<n; i++) {
            v1 += yx[jn+i]*d1[i];
            v2 += x2[jn+i]*d2[i];
            if(d2[i] > 0) pct += 1.0;
          }
          pct /= n; // percentage of residuals with absolute values below gamma
          if (pct < 0.05 || pct < 1.0/n || v2 == 0.0) {
            // approximate v2 with a continuation technique
            for (i=0; i<n; i++) {
              tmp = fabs(r[i]);
              if (tmp > gamma) v2 += x2[jn+i]/tmp;
            }
          }
          v1 = (v1+syx[j])/(2.0*n); v2 /= 2.0*n;
          // Update w_j
          w[j] = w_old[j] + v1/v2; 
          // Update r, d1, d2 and compute candidate of max_update
          change = w[j]-w_old[j];
          if (fabs(change) > 1e-6) {
            for (i=0; i<n; i++) {
              r[i] -= yx[jn+i]*change;
              if (fabs(r[i]) > gamma) {
                d1[i] = sign(r[i]);
                d2[i] = 0.0;
              } else {
                d1[i] = r[i]*gi;
                d2[i] = gi;
              }
            }
            update = v2*change*change;
            if (update > max_update) max_update = update;
            w_old[j] = w[j];
          }
          if (update < thresh) break;
        }
      }
    }
    // Check for convergence
    if (max_update < thresh) break;
  }
}