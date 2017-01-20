// [[Rcpp::depends(bigmemory, BH)]]
#include <Rcpp.h>
#include <bigmemory/MatrixAccessor.hpp>
#include <assert.h>

using namespace Rcpp;


template<typename T>
class SubMatrixAccessor {
public:
  typedef T value_type;
  
public:
  SubMatrixAccessor(BigMatrix &bm,
                    const IntegerVector &row_ind,
                    const NumericMatrix &covar) {
    if (covar.nrow() != 0) {
      assert(bm.nrow() == covar.nrow());
      _ncoladd = covar.ncol();
      _covar = covar;
    }  else {
      _ncoladd = 0;
    }
    
    int n = row_ind.size();
    std::vector<index_type> row_ind2(n);
    for (int i = 0; i < n; i++)
      row_ind2[i] = static_cast<index_type>(row_ind[i]);
    
    _pMat = reinterpret_cast<T*>(bm.matrix());
    _totalRows = bm.total_rows();
    _row_ind = row_ind2;
    _nrow = row_ind.size();
    _ncolBM = bm.ncol();
  }
  
  inline double operator() (int i, int j) {
    if (j < _ncolBM) {
      return *(_pMat + _totalRows * j + _row_ind[i]);
    } else {
      return _covar(i, j - _ncolBM);
    }
  }
  
  int nrow() const {
    return _nrow;
  }
  
  int ncol() const {
    return _ncolBM + _ncoladd;
  }
  
protected:
  T *_pMat;
  index_type _totalRows;
  int _nrow;
  int _ncolBM;
  std::vector<index_type> _row_ind;
  int _ncoladd;
  NumericMatrix _covar;
};
