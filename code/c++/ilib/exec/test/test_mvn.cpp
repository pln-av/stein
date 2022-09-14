#include <iostream>
#include <cmath>
#include <numbers>

#include "mkl.h"
#include "mathimf.h"
#include "stein/util/types.H"
#include "stein/la/matrix.H"
#include "stein/la/lapacke.H"
#include "stein/la/util.H"
#include "stein/la/blas.H"

#include "stein/math/target.H"

int main()
{
  std::cout << "Test Distribution Classes." << std::endl;

  stein::util::int_t d { 2 };
  stein::la::RowVector<float> mu(d, 0.0);
  mu(0) = -3.14;
  mu(1) = 3.14;

  stein::la::Matrix<float> SigmaSq(d, d);
  SigmaSq(0, 0) = 2.0;
  SigmaSq(1, 1) = 1.0;
  SigmaSq(0, 1) = SigmaSq(1, 0) = 0.50;

  std::cout << "mean: \n" << mu << std::endl;
  std::cout << "sigmasq: \n" << SigmaSq << std::endl;

  
  stein::target::MVN mvn(mu, SigmaSq);

  stein::la::RowVector<float> x(2);
  x(0) = -2.0; x(1) = 1.0;
  std::cout << mvn.pdf(x) << std::endl;
  
  /*
  stein::la::Matrix<float> X(2, 2);
  stein::la::RowVector<float> pdf(2);
  X(0, 0) = -1.0; X(0, 1) = 2.5;
  X(1, 0) = -3.2; X(1, 1) = 3.0;
  mvn.pdf(X, pdf);
  std::cout << "pdf: \n" << pdf << std::endl;
  */
  
  float nu { 2.21 };
  stein::target::UVT t { nu };

  /*std::cout << t.pdf( 1.0 ) << std::endl;
  std::cout << t.pdf( -1.0 ) << std::endl;
  std::cout << t.pdf( 0.25 ) << std::endl;*/
  

  
}
