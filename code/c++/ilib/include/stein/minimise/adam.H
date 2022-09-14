#pragma once

#include <iostream>
#include <cmath>
#include <numeric>

#include "stein/util/io.H"
#include "stein/util/types.H"
#include "stein/minimise/types.H"

namespace stein {
  namespace minimise {

    /*

      This is commented out for now, since I made changes in nesterov and can't be bothered updating here.
      Also, a part of the update is incorrect - see my python code in exec/test/opt 

    // standard gradient descent
    template <typename FunctionType, typename Criteria>
    requires ZeroOrderMinimiser<FunctionType> && TerminationCriteria<Criteria>
    class Adam
    {
    public:
      using ValueType = typename FunctionType::ValueType;
      using ParameterType = typename FunctionType::ParameterType;
      Adam(FunctionType f, Criteria tc, util::Writer& writer) :
	_function(f),
	_tc(tc),
	_dimension(f.dimension()),
	_grad(f.dimension()),
	_grad_sq(f.dimension()),
	_v(f.dimension()),
	_x(f.dimension()),
	_writer(writer){}

      void minimise(const ParameterType& ic, ParameterType& out)
      {
	std::cout << "x0: " << ic << std::endl;
	
	stein::util::int_t idx { 1 };
	ValueType abs_tol { std::numeric_limits<ValueType>::max() };
	ValueType rel_tol { std::numeric_limits<ValueType>::max() };
	ValueType grad_tol { std::numeric_limits<ValueType>::max() };
	ValueType f_prev, f_curr;

	// initialisations
	_writer.initialise();

	// initialisations
	const ValueType _zero { static_cast<ValueType>(0.0) };
	const ValueType _alpha { 0.001 };
	const ValueType _gamma_v { 0.9 };
	const ValueType _gamma_s { 0.999 };

	const ValueType _small_gamma_v { 1.0E-6 };
	const ValueType _small_gamma_s { 1.0E-6 };
	ValueType _gamma_v_k { _gamma_v };
	ValueType _gamma_s_k { _gamma_s };
	
	// initial condition
	std::copy(ic.begin(), ic.end(), _x.begin());
	_function.gradient(_x, _grad);
	std::fill(_v.begin(), _v.end(), _zero);
	std::fill(_grad_sq.begin(), _grad_sq.end(), _zero);

	f_prev = _function(_x);
	std::cout << "Initial Condition" << std::endl;
	std::cout << " -- x_{0}: " << _x << std::endl;
	std::cout << " -- v_{0}: " << _v << std::endl;
	std::cout << " -- g_{0}: " << _grad << std::endl;
	std::cout << " -- f_{0}: " << f_prev << std::endl;
	std::cout << " -------------------------------------- " << std::endl;

	f_prev = _function(out);
	_writer.write(_x, f_prev);
	while ( !_tc.stop(abs_tol, rel_tol, grad_tol, idx) )
	  {

	    std::cout << "Iteration(" << idx << ")" << std::endl;
	    std::cout << " -- x_{k}: " << _x << std::endl;
	    std::cout << " -- v_{k}: " << _v << std::endl;
	    std::cout << " -- g_{k}: " << _grad << std::endl;
	    std::cout << " -- gamma_{k}: " << _gamma_v_k << "," << _gamma_s_k << std::endl;
	    _gamma_v_k =
	      std::max(_gamma_v*_gamma_v_k, _small_gamma_v);
	    _gamma_s_k = std::max(_gamma_s*_gamma_s_k, _small_gamma_s);
	    //_gamma_v_k = _gamma_v *_gamma_v_k;
	    //_gamma_s_k = _gamma_s *_gamma_s_k;
	    for (stein::util::int_t j=0; j<_dimension; ++j)
	      {
		const ValueType gk_j { _grad(j) };

		// update velocity and grad square
		// Eqn 5.29, 5.30
		_v(j) = _gamma_v*_v(j) + (1-_gamma_v)*gk_j;
		_grad_sq(j) = _gamma_s*_grad_sq(j) + (1-_gamma_s)*gk_j*gk_j;

		// Eqn 5.31, 5.32
		_v(j) = _v(j)/(1.0 - _gamma_v_k);
		_grad_sq(j) = _grad_sq(j)/(1.0 - _gamma_s_k);

		// Eqn 5.33
		_x(j) = _x(j) - _alpha*_v(j) / ( _eps + std::sqrt(_grad_sq(j)) );
	      }

	    // update is complete.
	    // set for next iteration and stopping condition
	    _function.gradient(_x, _grad);
	    f_curr = _function(_x);
	    abs_tol = std::abs(f_curr - f_prev);
	    rel_tol = abs_tol / f_prev;
	    grad_tol = 0.0;
	    for (stein::util::int_t j=0; j<_dimension; ++j) { grad_tol += _grad(j)*_grad(j); }
	    grad_tol = std::sqrt(grad_tol);
	    _writer.write(_x, f_curr);
	    f_prev = f_curr;
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
      ParameterType _grad_sq;
      ParameterType _v;
      ParameterType _x;
      ValueType _eps { 1.0E-8 };
      util::Writer& _writer;
    };
    */
  }
}
