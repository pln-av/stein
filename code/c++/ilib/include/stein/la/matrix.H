/*
 * Simple matrix class for use with intel MKL/DAAL
 * Note that internal data is public, for ease of use at expense of security.
 */

#pragma once

#include <iostream>
#include <string>
#include <cassert>

#include "mkl.h"

#include "stein/util/types.H"



namespace stein {
  namespace la {

    template <typename T>
    class Matrix
    {

      /*
       * basic matrix type for stein:: project
       * basic info:
       *  --> design based around use in intel mkl
       *  --> size types all stein::util::int_t aka MKL_INT to match use in mkl
       *  --> 64 bit aligned and row ordered
       */


    public:
      const stein::util::int_t align { 64 };
      using value_type = T;
      
      ~Matrix(void)
      {
	// works when _data is nullptr
	mkl_free(_data);
      }
   
      Matrix(stein::util::int_t rows, stein::util::int_t cols, T t = T()) : _rows(rows), _cols(cols), _size(rows*cols)
      {
	// check for size errors
	assert( _size_check(rows, cols) && "sizes must be both zero, or both non-zero." );

	// memory only if size>0
	if (!empty())
	  {
	    _data = static_cast<T*>( mkl_calloc(_size, sizeof(T), align) );
	    std::fill(_data, _data+_size, t);
	  }
      }
      Matrix(void) : Matrix(0, 0) { std::cout << "Default" << std::endl; }

      Matrix(const Matrix& m) : _cols(m.cols()), _rows(m.rows()), _size(m.size())
      {
	// deep copy
	if (!empty())
	  {
	    _data = static_cast<T*>( mkl_calloc(_size, sizeof(T), align) );
	    std::copy(m.begin(), m.end(), _data);
	  }
      }

      Matrix& operator=(const Matrix& m)
      {
	// must assign to matrix with same dimension
	auto&& [r, c] = m.dimension();
	assert(r==rows() && c==cols() && "dimensions must be equal for asignment." );
	std::copy(m.begin(), m.end(), _data);
	return *this;
      }

      Matrix(Matrix&& m)
      {
	_data = m._data;
	_cols = m.cols();
	_rows = m.rows();
	_size = m.size();

	// destroy
	m._data = nullptr;
   
      }

      Matrix& operator=(Matrix&& m)
      {
	assert(m.rows()==rows() && m.cols()==cols() && "dimension must be equal for assignment." );
	_data = m._data;
	_cols = m.cols();
	_rows = m.rows();
	_size = m._size;

	// destrory
	m._data = nullptr;
	return *this;
      }
  
    public:

      // const and non-const access to data
      inline T* const data(void) const { return _data; }
      inline T* const data(void) { return _data; }

      // const iterators
      inline const T* const begin(void) const { return _data; }
      inline const T* const end(void) const { return _data+_size; }

      // non-const iterators, but protect pointer location
      inline T* const begin(void) { return _data; }
      inline T* const end(void) { return _data + _size; }

      // size data 
      inline stein::util::int_t cols(void) const { return _cols; }
      inline stein::util::int_t rows(void) const { return _rows; }
      inline bool empty(void) const { return _size==0; }
      inline stein::util::int_t size(void) const { return _size; }
      inline std::pair<stein::util::int_t, stein::util::int_t> dimension(void) const { return {rows(), cols()}; };
      
      // shape
      inline bool is_vector(void) const { return (_cols == 1) || (_rows == 1); }
      inline bool is_matrix(void) const { return (_cols > 1) && (_rows > 1); }
      inline bool is_square(void) const { return _cols == _rows; }
      
      // access as matrix
      inline T operator()(stein::util::int_t r, stein::util::int_t c) const { return _data[_cols*r + c]; }
      inline T& operator()(stein::util::int_t r, stein::util::int_t c) { return _data[_cols*r + c]; }

      // access as raw data so row/column vectors can be used in blas.  also add []
      // so that these work where a c array/ptr is expected
      inline T operator[](stein::util::int_t i) const { return _data[i]; }
      inline T& operator[](stein::util::int_t i) { return _data[i]; }
      inline T operator()(stein::util::int_t i) const { return _data[i]; }
      inline T& operator()(stein::util::int_t i)  { return _data[i]; }
      
      
  
    public:

      friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& m)
      {
	if (m.empty()) { os << "()"; return os; }
	const stein::util::int_t r { m.rows() };
	const stein::util::int_t c { m.cols() };
	for (stein::util::int_t i=0; i<r; ++i)
	  {
	    std::string s {"("};
	    for (stein::util::int_t j=0; j<c; ++j)
	      {
		s += std::to_string( m(i, j) );
		if (j < c-1) { s += ","; }
	      }
	    s += ")";
	    if (i < r-1) { s += "\n"; }
	    os << s;
	  }
	return os;
      }
 
  
    private:

      // data
      // these guys are not resizable
      const stein::util::int_t _rows {0};
      const stein::util::int_t _cols {0};
      const stein::util::int_t _size {0};
      T* _data { nullptr };

    private:
      bool _size_check(stein::util::int_t r, stein::util::int_t c)
      {

	// if size is zero, both r and c must be zero
	const stein::util::int_t s { r*c };
    
	if ( (s==0) and (r>0) ) { return false; }
	if ( (s==0) and (c>0) ) { return false; }
	return true;
      }
    };
  
    template <typename T>
    class ColumnVector : public Matrix<T>
    {
    public:
      ColumnVector(stein::util::int_t r, T t=T()) : Matrix<T>(r, 1, t) {};
    };


    template <typename T>
    class RowVector : public Matrix<T>
    {
    public:
      RowVector(stein::util::int_t c, T t=T()) : Matrix<T>(1, c, t) {};
    };

    // some type traits to use throughout
    template <typename T>
    struct is_matrix
    {
      static constexpr bool value = false;
    };

    template <typename T>
    struct is_matrix<Matrix<T>> { static constexpr bool value = true; };

    template <typename T>
    constexpr bool is_matrix_v = is_matrix<T>::value;
    
    template <typename T>
    struct is_vector
    {
      static constexpr bool value = false;
    };

    template <typename T>
    struct is_vector<ColumnVector<T>> { static constexpr bool value = true; };

    template <typename T>
    struct is_vector<RowVector<T>> { static constexpr bool value = true; };

    template <typename T>
    constexpr bool is_vector_v = is_vector<T>::value;
    
    template <typename T>
    concept IsFloatMatrix = is_matrix_v<T> && stein::util::is_mkl_float_v<typename T::value_type>;

    template <typename T>
    concept IsFloatVector = is_vector_v<T> && stein::util::is_mkl_float_v<typename T::value_type>;

  }
}
