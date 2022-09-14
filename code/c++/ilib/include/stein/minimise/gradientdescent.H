#pragma once

#include <iostream>
#include <cmath>

#include "stein/util/io.H"
#include "stein/util/types.H"
#include "stein/minimise/types.H"

namespace stein {
  namespace minimise {

    /*
      This is commented out, because I made chanes to nesterov that i haven't bothered
      to update here.  see my python code in exec/test/opt for some details re changes.
    
    // standard gradient descent
    template <typename FunctionType, typename Criteria>
    requires ZeroOrderMinimiser<FunctionType> && TerminationCriteria<Criteria>
    class GradientDescent
    {
    public:
      using ValueType = typename FunctionType::ValueType;
      using ParameterType = typename FunctionType::ParameterType;
      GradientDescent(FunctionType f, Criteria tc, util::Writer& writer, ValueType gamma)
	: _function(f), _tc(tc), _dimension(f.dimension()), _grad(f.dimension()), _writer(writer), _gamma(gamma){}
      void minimise(const ParameterType& ic, ParameterType& out)
      {
	std::cout << "Begin GD Minimisation." << std::endl;
	stein::util::int_t idx { 0 };
	ValueType abs_tol { std::numeric_limits<ValueType>::max() };
	ValueType rel_tol { std::numeric_limits<ValueType>::max() };
	ValueType grad_tol { std::numeric_limits<ValueType>::max() };
	ValueType f_prev, f_curr;

	_writer.initialise();
	std::copy(ic.data(), ic.data()+_dimension, out.data());
	while ( !_tc.stop(abs_tol, rel_tol, grad_tol, idx) )
	  {
	
	    // compute gradient vector.
	    // out := x_k
	    // _tmp_x := g_k
	
	    f_prev = _function(out);
	    _writer.write(out, f_prev);
	    _function.gradient(out, _grad);

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
	    f_curr = _function(out);
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
      const ValueType _gamma;
      util::Writer& _writer;
    };
    */
  }
}
