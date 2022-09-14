#pragma once
#include "stein/util/types.H"
#include "stein/la/matrix.H"

namespace stein {
  namespace la {
    namespace blas {
      static constexpr stein::util::int_t inc { 1 };
      static constexpr CBLAS_LAYOUT layout { CblasRowMajor };
      using int_t = stein::util::int_t;
      using Matrix_i = stein::la::Matrix<int_t>;
      using Matrix_f = stein::la::Matrix<float>;
      using Matrix_d = stein::la::Matrix<double>;

      template <typename T, typename U>
      requires IsFloatMatrix<T> && IsFloatVector<U> && std::is_same_v<typename T::value_type, typename U::value_type>
      void symv(const T& A, const U& x, U& y, typename T::value_type alpha=1.0, typename T::value_type beta=0.0, CBLAS_UPLO uplo=CblasLower)
      {
	// compute y := alpha*A*x + beta*y where A is symmetric
	if constexpr (stein::util::is_float_v<typename T::value_type>)
	   cblas_ssymv(layout, uplo, A.rows(), alpha, A.data(), A.rows(), x.data(), inc, beta, y.data(), inc);
	if constexpr (stein::util::is_double_v<typename T::value_type>)
	   cblas_dsymv(layout, uplo, A.rows(), alpha, A.data(), A.rows(), x.data(), inc, beta, y.data(), inc);
      }

      template <typename T, typename U>
      requires std::is_same_v<typename T::value_type, typename U::value_type> && stein::util::is_mkl_float_v<typename T::value_type>
      typename T::value_type dot(const T& x, const U& y)
      {
	// ordinarily this is defined for vectors, though it could be for matrices too.
	// here I allow T, U to be different, though they should have same value_type and of course size
	if constexpr (stein::util::is_float_v<typename T::value_type>)
	   return cblas_sdot(x.size(), x.data(), inc, y.data(), inc);
	if constexpr (stein::util::is_double_v<typename T::value_type>)
	   return cblas_ddot(x.size(), x.data(), inc, y.data(), inc);
      }

      template <typename T>
      requires stein::util::is_mkl_float_v<typename T::value_type>
      typename T::value_type norm2(const T& x)
      {
	if constexpr (stein::util::is_float_v<typename T::value_type>)
	   return cblas_snrm2(x.size(), x.begin(), inc);
	if constexpr (stein::util::is_double_v<typename T::value_type>)
	   return cblas_dnrm2(x.size(), x.begin(), inc);
      }
    }
  }
}
