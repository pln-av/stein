#pragma once

#include <iostream>
#include <cmath>
#include <numeric>

#include "stein/util/io.H"
#include "stein/util/types.H"
#include "stein/minimise/types.H"

namespace stein {
  namespace minimise {

    // standard gradient descent
    template <typename FunctionType, typename Criteria>
    requires ZeroOrderMinimiser<FunctionType> && TerminationCriteria<Criteria>
    class HyperGradientDescent
    {
    public:
      using ValueType = typename FunctionType::ValueType;
      using ParameterType = typename FunctionType::ParameterType;
      HyperGradientDescent(FunctionType f, Criteria tc, util::Writer& writer, ValueType gamma, ValueType beta) :
	_function(f),
	_tc(tc),
	_dimension(f.dimension()),
	_grad(f.dimension()),
	_prev_grad(f.dimension()),
	_writer(writer),
	_gamma(gamma),
	_beta(beta){}
      void minimise(const ParameterType& ic, ParameterType& out)
      {
	std::cout << "x0: " << ic << std::endl;
	
	stein::util::int_t idx { 0 };
	ValueType abs_tol { std::numeric_limits<ValueType>::max() };
	ValueType rel_tol { std::numeric_limits<ValueType>::max() };
	ValueType grad_tol { std::numeric_limits<ValueType>::max() };
	ValueType f_prev, f_curr;

	// initialisations
	_writer.initialise();

	// initialisations
	const ValueType _zero { static_cast<ValueType>(0.0) };
	std::copy(ic.begin(), ic.end(), out.data());
	std::fill( _grad.begin(), _grad.end(), _zero);
	std::fill(_prev_grad.begin(), _prev_grad.end(), _zero);

	std::cout << "Initial Condition" << std::endl;
	std::cout << " -- x_{0}: " << out << std::endl;
	std::cout << " -- g_{0}: " << _grad << std::endl;
	std::cout << " -- g_{-1}: " << _prev_grad << std::endl;
	std::cout << " -- f_{k}: " << _function(out) << std::endl;
	std::cout << " -------------------------------------- " << std::endl;
	ValueType _min_gamma { _gamma };
	while ( !_tc.stop(abs_tol, rel_tol, grad_tol, idx) )
	  {

	    f_prev = _function(out);
	    _writer.write(out, f_prev);

	    std::cout << "Iteration(" << idx+1 << ")" << std::endl;
	    std::cout << " -- x_{k}: " << out << std::endl;
	    // step 1. copy _grad to _prev_grad, then update _grad
	    std::copy(_grad.begin(), _grad.end(), _prev_grad.begin());
	    _function.gradient(out, _grad);
	    
	    std::cout << " -- g_{k}: " << _grad << std::endl;
	    std::cout << " -- g_{k-1}: " << _prev_grad << std::endl;

	    // step 2. hyperparameter update
	    const ValueType gamma_adj { _beta*std::inner_product( _grad.begin(), _grad.end(), _prev_grad.begin(), _zero) };
	    _gamma += gamma_adj;
	    std::cout << " -- gamma_{k}: " << _gamma << std::endl;

	    // step 3.  standard GD update using new learning rate _gamma

	    // do this update manually first
	    ValueType g_norm { 0.0 };
	    for (stein::util::int_t j=0; j<_dimension; ++j)
	      {

		// current x_k, g_k
		const ValueType gk_i { _grad(j) };
		g_norm += gk_i*gk_i;

		// do updates
		out(j) -= _gamma*gk_i;
	      }

	    std::cout << " -- x_{k+1}: " << out << std::endl;

	    f_curr = _function(out);
	    std::cout << " -- f_{k+1}: " << f_curr << std::endl;
	    const ValueType df { f_curr - f_prev };
	
	    // prepare stopping criteria
	    abs_tol = std::abs(df);
	    rel_tol = std::abs(df/f_curr);
	    grad_tol = std::sqrt(g_norm);
	    ++idx;
	  };

	_writer.write(out, f_curr);
	return;
      };
    private:

  
    private:
      const FunctionType _function;
      const Criteria _tc;
      const stein::util::int_t _dimension;
      ParameterType _grad;
      ParameterType _prev_grad;
      ValueType _gamma;
      ValueType _beta;
      util::Writer& _writer;
    };
  }
}
