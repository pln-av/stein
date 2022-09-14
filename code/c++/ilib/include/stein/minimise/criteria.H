#pragma once

#include "stein/minimise/types.H"

namespace stein {
  namespace minimise {
    template <typename T>
    struct StandardCriteria
    {
      using ValueType = typename T::ParameterType::value_type;
      using FunctionType = T;
      StandardCriteria(ValueType ma=1.0E-6, ValueType mr=1.0E-5, ValueType mg=1.0E-5, stein::util::int_t mi = 100)
	: max_abs_tol( ma ), max_rel_tol( mr ), max_iterations( mi ) {}

      // required data
      ValueType max_abs_tol;
      ValueType max_rel_tol;
      ValueType max_grad_tol;
      stein::util::int_t max_iterations;

      // required termination criteria
      bool stop(ValueType abs_tol, ValueType rel_tol, ValueType grad_tol, stein::util::int_t idx) const
      {
	const bool iters_stop { idx > max_iterations };
	const bool abs_tol_stop { abs_tol < max_abs_tol };
	const bool rel_tol_stop { rel_tol < max_rel_tol };
	const bool grad_stop { grad_tol < max_grad_tol };
	return iters_stop || abs_tol_stop || rel_tol_stop || grad_stop;
      }
    };

    template <typename T>
    struct GradientCriteria
    {
      using ValueType = typename T::ParameterType::value_type;
      using FunctionType = T;
      GradientCriteria(ValueType mg=1.0E-6, stein::util::int_t mi = 100)
	: max_grad_tol( mg ), max_iterations( mi ) {}

      // required data
      ValueType max_abs_tol;
      ValueType max_rel_tol;
      ValueType max_grad_tol;
      stein::util::int_t max_iterations;

      // required termination criteria
      bool stop(ValueType abs_tol, ValueType rel_tol, ValueType grad_tol, stein::util::int_t idx) const
      {
	const bool iters_stop { idx > max_iterations };
	const bool grad_stop { grad_tol < max_grad_tol };
	return iters_stop || grad_stop;
      }
    };
  }
}
