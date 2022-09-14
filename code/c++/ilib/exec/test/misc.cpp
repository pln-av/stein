#include <algorithm>
#include <iostream>
#include <vector>

int main()
{
  double div { 2 };
  std::vector<double> x { 1, 2, 3 };
  std::for_each(x.begin(), x.end(), [div](double& k) { k/=div; });
  for (auto&& xi : x) { std::cout << "xi: " << xi << std::endl; }
}
