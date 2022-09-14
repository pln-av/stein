#pragma once

#include "stein/util/types.H"
#include "stein/la/matrix.H"

namespace stein {
  namespace la {
    namespace util {

      enum class ZERO_TYPE { LOWER, UPPER, DIAGONAL, ALL };
      template <typename T>
      void zero(stein::la::Matrix<T>& m, ZERO_TYPE zt)
      {

	// all these assume row major
	const stein::util::int_t r { m.rows() };
	const stein::util::int_t c { m.cols() };
	if (zt==ZERO_TYPE::LOWER)
	  {
	    for (stein::util::int_t i=0; i<r; ++i)
	      {
		for (stein::util::int_t j=0; j<i; ++j)
		  { m(i, j) = static_cast<T>(0); }
	      }
	  }
	else if (zt==ZERO_TYPE::UPPER)
	  {
	    for (stein::util::int_t i=0; i<r; ++i)
	      {
		for (stein::util::int_t j=i+1; j<c; ++j)
		  { m(i, j) = static_cast<T>(0); }
	      }
	  }
	else if (zt==ZERO_TYPE::DIAGONAL)
	  {
	    const stein::util::int_t k = r < c ? r : c;
	    for (stein::util::int_t i=0; i<k; ++i)
	      { m(i, i) = static_cast<T>(0); } 
	  }
	else if (zt==ZERO_TYPE::ALL)
	  {
	    std::fill(m.begin(), m.end(), static_cast<T>(0) );
	  }
	else
	  {
	    std::cout << "unknown ZERO_TYPE." << std::endl;
	    abort();
	  }
      }

      template <typename T>
      T det_diag(const stein::la::Matrix<T>& m)
      {
	T mult { static_cast<T>(1.0) };
	const stein::util::int_t n { m.rows() };
	for (stein::util::int_t i=0; i<n; ++i) { mult *= m(i, i); }
	return mult;
      }

    }
  }
}
