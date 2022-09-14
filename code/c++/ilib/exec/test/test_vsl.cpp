#include <iostream>
#include <cmath>
#include <stdlib.h>

#include "mkl.h"
#include "stein/la/matrix.H"
#include "stein/la/util.H"
#include "stein/la/lapacke.H"

template <typename T>
void moments(const stein::la::Matrix<T>& r)
{
  // compute moments of rv generated to check
  // assume there is two dimensions
  const stein::util::int_t n { r.rows() };
  const stein::util::int_t d { r.cols() };
  //std::cout << r << std::endl;
  
  double sum_x {0.0}; double sumsq_x {0.0};
  double sum_y {0.0}; double sumsq_y {0.0};
  double sum_xy {0.0};
  for (stein::util::int_t i=0; i<n; ++i)
    {
      const double xi { r(i, 0) };
      const double yi { r(i, 1) };
      sum_x += xi;
      sum_y += yi;
      
      sumsq_y += yi*yi;
      sumsq_x += xi*xi;
      
      sum_xy += xi*yi;
      
      //std::cout << yi << std::endl;
    }

  const double denom = 1.0/static_cast<double>(n-1.0);
  // means
  double mean_x { sum_x / n };
  double mean_y { sum_y / n};

  // variance
  double sig_x { denom*(sumsq_x - n*mean_x*mean_x) };
  double sig_y { denom*(sumsq_y - n*mean_y*mean_y) };

  // covariance
  double cov_xy { denom*(sum_xy - n*mean_x*mean_y) };
  /*
  
  double mean_y { sum_y / n};
  double sig_y { (sumsq_y - n*mean_y*mean_y)/(n-1) };
  double cov_xy { (sum_xy - n*mean_x*mean_y) / (n-1) };
  */
  std::cout << "mean(x): " << mean_x << "| var(x): " << sig_x << std::endl;
  std::cout << "mean(y): " << mean_y << "| var(y): " << sig_y << std::endl;
  std::cout << "cov(x,y): " << cov_xy << std::endl;
  
  
}

int main()
{

  std::cout << "Test vsl" << std::endl;

  const stein::util::int_t method { VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER2 };

  // problem size and set up
  const stein::util::int_t n { 1000 };
  const stein::util::int_t d { 2 };

  // a is mean vector dimension 2
  stein::la::ColumnVector<double> a(2, 0.0);
  a(0) = 30.14; a(1) = -30.14;
  std::cout << "Mean: " << std::endl;
  std::cout << a << std::endl;
  
  stein::la::Matrix<double> t(2, 2);
  const stein::util::int_t mstorage { VSL_MATRIX_STORAGE_FULL };
  t(0, 0) = 0.50;
  t(1, 1) = 2.0;
  t(0, 1) = t(1, 0) = -0.50;
  std::cout << "covariance: " << std::endl;
  std::cout << t << std::endl;
  stein::la::lapacke::cholesky(t, 'L');
  std::cout << "cholesky(covariance):" << std::endl;
  std::cout << t << std::endl;

  // setup the stream
  VSLStreamStatePtr stream;
  srand(time(NULL));
  
  stein::util::int_t err { vslNewStream(&stream, VSL_BRNG_MT2203, 10) } ;
  //CheckVslError(err);
  
  stein::la::Matrix<double> r(n, d);


  stein::util::int_t status { vdRngGaussianMV(method, stream, r.rows(), r.data(), r.cols(), mstorage, a.data(), t.data()) };
  if (status != VSL_STATUS_OK)
    {
      std::cout << "VSL status in error state: " << status << std::endl; 
    }
  moments(r);


  
}
