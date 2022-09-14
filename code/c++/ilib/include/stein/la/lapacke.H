#pragma once

#include "stein/util/types.H"
#include "stein/la/matrix.H"

#include <iostream>

namespace stein {
  namespace la {
    namespace lapacke {
      static constexpr stein::util::uint_t layout { MKL_ROW_MAJOR };
      static constexpr stein::util::uint_t trans { MKL_NOTRANS };
      // compute cholesky decompositions in place

      using int_t = stein::util::int_t;
      using Matrix_i = stein::la::Matrix<int_t>;
      using Matrix_f = stein::la::Matrix<float>;
      using Matrix_d = stein::la::Matrix<double>;

      // interfaces to float/double calls
      template <typename T>
      requires IsFloatMatrix<T>
      int_t cholesky(T& t, char uplo='L')
      {

	// compute cholesky factorisation and store in uplo
	if constexpr ( stein::util::is_float_v<typename T::value_type> )
	   return LAPACKE_spotrf(layout, uplo, t.rows(), t.data(), t.cols()); 
	if constexpr ( stein::util::is_double_v<typename T::value_type> )
	   return LAPACKE_dpotrf(layout, uplo, t.rows(), t.data(), t.cols()); 
      };

      template <typename T>
      requires IsFloatMatrix<T>
      int_t inv_cholesky(T& t, char uplo='L')
      {

	// compute inverse of matrix t using cholesky
	// NB t should be cholesky factorised first before calling here.
	// NB uplo describes if t was stored in L or U
	if constexpr ( stein::util::is_float_v<typename T::value_type> )
	   return LAPACKE_spotri( LAPACK_ROW_MAJOR, uplo, t.rows(), t.data(), t.rows());
	if constexpr ( stein::util::is_double_v<typename T::value_type> )
	   return LAPACKE_dpotri( LAPACK_ROW_MAJOR, uplo, t.rows(), t.data(), t.rows());
      }

      template <typename T>
      requires IsFloatMatrix<T>
      int_t lu(T& t, Matrix_i& ipiv)
      {
	// compute lu factorisation (partial pivoting) 
	if constexpr ( stein::util::is_float_v<typename T::value_type> )
	   LAPACKE_sgetrf(layout, t.rows(), t.cols(), t.data(), t.cols(), ipiv.data());
	if constexpr ( stein::util::is_double_v<typename T::value_type> )
	   LAPACKE_dgetrf(layout, t.rows(), t.cols(), t.data(), t.cols(), ipiv.data());
      }

      template <typename T>
      requires IsFloatMatrix<T>
      int_t inv_lu(T& t, Matrix_i& ipiv)
      {
	// compute inverse of matrix t using lu.
	// NB t should be lu factorised first before calling here.  pivots stored in ipiv
	if constexpr ( stein::util::is_float_v<typename T::value_type> )
	   LAPACKE_sgetri(layout, t.rows(), t.data(), t.rows(), ipiv.data());
	if constexpr ( stein::util::is_double_v<typename T::value_type> )
	   LAPACKE_dgetri(layout, t.rows(), t.data(), t.rows(), ipiv.data());
      }
    }
  }
}
