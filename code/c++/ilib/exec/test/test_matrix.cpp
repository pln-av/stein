#include <iostream>
#include <string>

#include "stein/la/matrix.H"
#include "stein/la/lapacke.H"
#include "stein/la/util.H"


int main()
{
  std::cout << "Test matrix.H" << std::endl;

  // lets try a call to [s|d]potrf, one of the MKL cholesky algorithms

  // create an example Covariance matrix
  std::cout << "Construct Matrix Sigma." << std::endl;
  stein::la::Matrix<float> Sigma(2, 2);
  Sigma(0, 0) = Sigma(1, 1) = 1.0;
  Sigma(0, 1) = Sigma(1, 0) = 0.5;
  std::cout << Sigma << std::endl;


  /*
  // compute the cholesky
  stein::la::Matrix<float> Chol(2, 2);
  const stein::util::uint_t layout { MKL_ROW_MAJOR };   // row or col major
  const stein::util::uint_t trans { MKL_NOTRANS };      // no transpose
  const char uplo { 'U' };                // upper/lower, but not important in this case

  // leading dimension
  // if X==(M, N) and X is row-major, then lda=N (ie number of columns)
  

  stein::la::ColumnVector v(4, 3.14);
  std::cout << v << std::endl;

  std::cout << v.is_matrix() << std::endl;
  std::cout << v.is_vector() << std::endl;
  std::cout << v[1] << std::endl;
  std::cout << Sigma[2] << std::endl;

  std::cout << Sigma(2) << std::endl;
  std::cout << v(1) << std::endl;

  
  std::cout << stein::la::is_matrix<stein::la::ColumnVector<float>>::value << std::endl;
  std::cout << stein::la::is_vector<stein::la::ColumnVector<float>>::value << std::endl;

  // lets try out new template stuff in lapacke.
  */
  std::cout << "Cholesky." << std::endl;
  auto info1 = stein::la::lapacke::cholesky(Sigma, 'L');
  std::cout << Sigma << std::endl;
  auto info2 = stein::la::lapacke::inv_cholesky(Sigma, 'L');
  std::cout << Sigma << std::endl;

  stein::la::RowVector<stein::util::int_t> ipiv(2);
  auto info3 = stein::la::lapacke::lu(Sigma, ipiv);
  auto info4 = stein::la::lapacke::inv_lu(Sigma, ipiv);
  std::cout << Sigma << std::endl;
}
