#pragma once

#include <cmath>
#include <numbers>

#include "mkl.h"
#include "stein/util/types.H"
#include "stein/la/matrix.H"
#include "stein/math/special.H"

namespace stein {
  namespace target {

    template <typename T, typename U>
    requires stein::la::IsFloatVector<T> && stein::la::IsFloatMatrix<U> && std::is_same_v<typename T::value_type, typename U::value_type>
    class MVN
    {
    public:
      using float_type = typename T::value_type;
      
      MVN(const T& mu, const U& Sigma)
	: _mu(mu), _err(mu), _tmp(mu), _invSigma(Sigma), _dimension(Sigma.rows())
      {
	// precomputed data required is
	// (1.) covariance matrix inverse
	// (2.) covariance matrix determinant
	// (3.) multiplicative constant

	// care must be taken with _invSigma matrix, which after cholesky is lower triangular only
	// it is symetric.  this is why symv is used for matrix-vector multiplication later.
	
	// NB to get determinant, we need to factorise.  can do lu or chol
	stein::la::lapacke::cholesky( _invSigma, 'L' );
	float_type _determinant { stein::la::util::det_diag( _invSigma ) };
	_determinant *= _determinant;

	const stein::util::int_t info { stein::la::lapacke::inv_cholesky( _invSigma, 'L') };
	assert(info==0);

	const float_type arg1 { static_cast<float_type>( std::pow( 2.0*std::numbers::pi_v<float_type>, _dimension ) ) };
	const float_type arg2 { static_cast<float_type>( arg1 * _determinant ) };
	_constant = static_cast<float_type>(1.0) / std::sqrt(arg2);
      };

      template <typename V>
      requires stein::la::IsFloatVector<V> && std::is_same_v<float_type, typename V::value_type>
      float_type pdf(const V& x)
      {
	// x is vector of size d with one observation and _dimension variables

	// compute err
	for (stein::util::int_t i=0; i<x.size(); ++i)
	  { _err(i) = x(i) - _mu(i); }
	
	// two step quadratic form
	stein::la::blas::symv(_invSigma, _err, _tmp);
	return _constant * std::exp( -0.5* stein::la::blas::dot(_err, _tmp) );
      }

      // TODO: add a fill matrix version of the method above
    private:
      T _mu;
      T _err;
      T _tmp;
      U _invSigma;
      
      stein::util::int_t _dimension;

      float_type _constant;
      
    };


    // Univariate T Distribution (standardised to have variance=1.0)
    template <typename T>
    requires stein::util::is_mkl_float_v<T>
    class UVT
    {
    public:
      UVT(T nu) : _nu(nu)
      {
	_scale = std::sqrt( (nu - static_cast<T>(2.0))/nu );
	_inv_scale = T(1.0)/_scale;
	
	const T num { stein::special::gamma( T(0.5)*(nu + T(1.0))) };
	const T den { std::sqrt( std::numbers::pi_v<T> * nu ) * stein::special::gamma( T(0.5*nu) ) };
	_constant = _inv_scale * num / den;
	
      }

      T pdf(T x)
      {
	const T indep_arg { x*_inv_scale };
	const T pow_arg { T(1) + indep_arg*indep_arg/_nu };
	const T pow_exp { T(-0.5)*(T(1.0) + _nu) };
	return _constant * std::pow(pow_arg, pow_exp);
      }
      
    private:

      // dof and multiplicative constant which can be precomputed
      const T _nu;

      T _constant;
      T _scale;
      T _inv_scale;
    };

  }
}
