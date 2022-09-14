#include <iostream>
#include <numeric>
#include <vector>
#include "stein/la/matrix.H"
#include "stein/minimise/minimise.H"

#include "mkl.h"

// use Risenbrock as test function
template <typename T>
class Rosenbrock
{
public:
  using ParameterType = T;
  using ValueType = typename T::value_type;
  Rosenbrock(ValueType kappa) : _kappa(kappa) {  }
  ValueType operator() (const ParameterType& x) const
  {
    // required to be a ZeroOrderFunction
    const ValueType x0 { x(0) };
    const ValueType x1 { x(1) };
    const ValueType arg { x1 - x0*x0 };
    return _kappa*arg*arg + (x0 - 1.0)*(x0 - 1.0);
    
  }

  void gradient(const ParameterType& x, ParameterType& g) const
  {
    const ValueType x0 { x(0) };
    const ValueType x1 { x(1) };
    
    const ValueType arg { x1 - x0*x0 };
    g(0) = -4.0*_kappa*x0*arg + 2.0*(x0 - 1.0);
    g(1) = 2.0*_kappa*arg;
    return;
  }
  stein::util::int_t dimension(void) const { return _dimension; }
  
private:
  const ValueType _kappa;
  const stein::util::int_t _dimension { 2 };
};

int main()
{

  std::cout << "Test Minimisation Code." << std::endl;

  std::cout << "Show how to switch types. " << std::endl;
  using ValueType = float;
  using ParameterType = stein::la::RowVector<ValueType>;

  // objective function
  ValueType kappa { 100.0 };
  Rosenbrock<ParameterType> rosenbrock {kappa};

  // solver with termination criteria
  ValueType gtol { 0.01 };
  stein::util::int_t maxiters { 10000 };
  stein::minimise::GradientCriteria<decltype(rosenbrock)> criteria{ gtol, maxiters };

  // nesterov parameters
  ValueType alpha { 0.20 };
  ValueType beta { 0.80 };
  ValueType gamma { 0.90 };
  stein::util::Writer writer { "opt/results.csv" };
  stein::minimise::NesterovParameters parameters { alpha, beta, gamma };
  stein::minimise::Nesterov nesterov { rosenbrock, criteria, parameters, writer };

  // set initial conditions
  ParameterType x(2), xmin(2);
  x(0) = -1.50;
  x(1) = 0.50;

  // solve and print some results
  auto&& results { nesterov.minimise(x, xmin) };
  std::cout << "Results: " << std::endl;
  std::cout << " -- compute time: " << results.time << std::endl;
  std::cout << " -- |df|: " << results.abs_ftol << std::endl;
  std::cout << " -- |df|/f: " << results.rel_ftol << std::endl;
  std::cout << " -- mean(|g|): " << results.abs_gtol << std::endl;
  std::cout << " -- iterations: " << results.n << std::endl;
  std::cout << " -- xic: " << x << std::endl;
  std::cout << " -- xopt: " << xmin << std::endl;


}
