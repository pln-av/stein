#include <iostream>

#include <autodiff/forward/dual.hpp>

autodiff::dual f(autodiff::dual x)
{
    return 1 + x + x*x + 1/x + log(x);
}

int main()
{

  std::cout << "Test." << std::endl;
  autodiff::dual x = 1.0;
  autodiff::dual u = f(x);
  double dudx = derivative(f, wrt(x), at(x));

  std::cout << "u = " << u << std::endl;
  std::cout << "du/dx = " << dudx << std::endl;
}
