#pragma once

#include "mkl.h"

namespace stein {
  namespace util {

    // integer types used in mkl routines
    using int_t = MKL_INT;
    using uint_t = MKL_UINT;

    // most mkl routines for float or double only (some long double, but ignore)
    // make this info available to simplify calling mkl routines
    
    // available mkl floats
    template <typename T>
    struct is_mkl_float
    {
      static constexpr bool value = false;
    };

    template <>
    struct is_mkl_float<float>
    {
      static constexpr bool value = true;
    };

    template <>
    struct is_mkl_float<double>
    {
      static constexpr bool value = true;
    };

    template <typename T>
    constexpr bool is_mkl_float_v = is_mkl_float<T>::value;


    template <typename T>
    struct is_double
    {
      static constexpr bool value = std::is_same<T, double>::value;
    };
    template <typename T>
    constexpr bool is_double_v = is_double<T>::value;

    template <typename T>
    struct is_float
    {
      static constexpr bool value = std::is_same<T, float>::value;
    };

    template <typename T>
    constexpr bool is_float_v = is_float<T>::value;

    // define Zero and First order functions
    // TODO: right now this wont work for single variable functions natively
    /* Concepts for ZeroOrderFunction and FirstOrderFunction */
    template <typename FunctionType>
    concept ZeroOrderFunction = requires (FunctionType f, typename FunctionType::ParameterType p)
      {
       typename FunctionType::ParameterType;
       typename FunctionType::ParameterType::value_type;
       { f(p) } -> std::same_as<typename FunctionType::ParameterType::value_type>;
       { f.dimension() } -> std::same_as<stein::util::int_t>;
      };

    template <typename FunctionType>
    concept FirstOrderFunction = requires (FunctionType f, typename FunctionType::ParameterType p)
      {
       requires ZeroOrderFunction<FunctionType>;
       { f.gradient(p, p) } -> std::same_as<void>;
      };

  }
}

