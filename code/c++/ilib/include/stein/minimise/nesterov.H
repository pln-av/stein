#pragma once

#include <iostream>
#include <cmath>
#include <numeric>

#include "mkl.h"

#include "stein/util/io.H"
#include "stein/util/types.H"
#include "stein/minimise/types.H"
#include "stein/la/blas.H"

namespace stein {
  namespace minimise {

    // nesterov minimiser
    template <typename T>
    struct NesterovParameters
    {
      NesterovParameters(T alpha, T beta, T gamma) : alpha(alpha), beta(beta), gamma(gamma) {}
      NesterovParameters(void) = default;
      T alpha { 0.0 };
      T beta { 0.0 };
      T gamma { 0.0 };
    };
    
    template <typename FunctionType, typename Criteria>
    requires ZeroOrderMinimiser<FunctionType> && TerminationCriteria<Criteria>
    class Nesterov
    {
      using ValueType = typename FunctionType::ValueType;
      using ParameterType = typename FunctionType::ParameterType;

    public:

      
      Nesterov(FunctionType f, Criteria tc, NesterovParameters<ValueType> parameters, util::Writer& writer)
	: _function(f),
	  _tc(tc),
	  _dimension(f.dimension()),
	  _g(f.dimension()),
	  _v(f.dimension()),
	  _x(f.dimension()),
	  _xhat(f.dimension()),
	  _writer(writer)
      {
	second();  // warm up this call
	_alpha = parameters.alpha;
	_beta = parameters.beta;
	_gamma = parameters.gamma;
      }
      MinimisationResults<ValueType> minimise(const ParameterType& ic, ParameterType& out)
      {

	const double t_start { dsecnd() };

	// iteration checks for termination criteria
	stein::util::int_t idx { 0 };
	ValueType abs_ftol { _max_val };
	ValueType rel_ftol { _max_val };
	ValueType abs_gtol { _max_val };

	// required initialisations
	_writer.initialise();
	std::copy(ic.begin(), ic.end(), _x.begin());
	std::fill(_v.begin(), _v.end(), _zero);
	std::fill(_g.begin(), _g.end(), _zero);
	
	// write initial conditions
	ValueType fprev = _function(_x);
	ValueType fcurr = _max_val;

	// hard-code these for now.
	//const ValueType alpha { 0.20 };
	//const ValueType beta { 0.80 };
	//const ValueType gamma { 0.90 };
	const ValueType _min_alpha { _min_alpha_frac*_alpha };
	ValueType alpha_k { _alpha };
	
	while ( !_tc.stop(abs_ftol, rel_ftol, abs_gtol, idx) )
	  {

	    // nesterov projection for x
	    for (stein::util::int_t j=0; j<_dimension; ++j)
	      {
		_xhat(j) = _x(j) + _beta*_v(j);
	      }

	    // compute normalised gradient, and do nesterov update
	    _function.gradient(_xhat, _g);
	    const ValueType gnorm { stein::la::blas::norm2(_g) };
	    const ValueType ignorm = 1.0/gnorm;
	    for (stein::util::int_t j=0; j<_dimension; ++j)
	      {
		_g(j) *= ignorm; 
		_v(j) = _beta*_v(j) - alpha_k*_g(j);
		_x(j) += _v(j);
	      }


	    // update results
	    fcurr = _function(_x);
	    _writer.write(_x, fcurr);

	    // prepare stopping criteria
	    const ValueType df { fcurr - fprev };
	    abs_ftol = std::abs(df);
	    rel_ftol = std::abs(df/fprev);
	    abs_gtol = gnorm/_dimension;

	    // prepare for next iteration
	    fprev = fcurr;
	    if (alpha_k > _min_alpha)
	      alpha_k *= _gamma;
	    ++idx;
	  };

	const double t_end { dsecnd() };	
	std::copy(_x.begin(), _x.end(), out.begin());
	return MinimisationResults<ValueType>  { t_end-t_start , abs_ftol, rel_ftol, abs_gtol, idx-1 };

      };
    private:

  
    private:
      const FunctionType _function;
      const Criteria _tc;
      const stein::util::int_t _dimension;
      ParameterType _g;
      ParameterType _v;
      ParameterType _x;
      ParameterType _xhat;
      util::Writer& _writer;

      ValueType _alpha;
      ValueType _beta;
      ValueType _gamma;
      const ValueType _min_alpha_frac { 0.01 };
      const ValueType _max_val { std::numeric_limits<ValueType>::max() };
      const ValueType _zero { static_cast<ValueType>(0.0) };
    };
  }
}
