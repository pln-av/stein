#pragma once

// defines basic types/concepts etc for use in minimisers
#include "stein/util/types.H"
#include "stein/la/matrix.H"

namespace stein {
  namespace minimise {

    template <typename T>
    struct MinimisationResults
    {
      MinimisationResults(void) = default;
      MinimisationResults(double time, T fatol, T frtol, T gtol, stein::util::int_t n) : time(time), abs_ftol(fatol), rel_ftol(frtol), abs_gtol(gtol), n(n) {}
      double time;
      T abs_ftol;
      T rel_ftol;
      T abs_gtol;
      stein::util::int_t n;
    };
    
    template <typename CriteriaType>
    concept TerminationCriteria = requires(CriteriaType ct, typename CriteriaType::FunctionType::ValueType v, stein::util::int_t mit)
      {
       requires stein::util::ZeroOrderFunction<typename CriteriaType::FunctionType>;

       // these member type checks are pretty gross!
       // see "concept member check type ambiguity with reference" on stackoverflow
       { ct.max_iterations } -> std::same_as<stein::util::int_t&>;
       { ct.max_abs_tol } -> std::same_as<typename CriteriaType::FunctionType::ValueType&>;
       { ct.max_rel_tol } -> std::same_as<typename CriteriaType::FunctionType::ValueType&>;
       { ct.max_grad_tol } -> std::same_as<typename CriteriaType::FunctionType::ValueType&>;
       { ct.stop(v, v, v, mit) } -> std::same_as<bool>;

      };

    template <typename FunctionType>
    concept ZeroOrderMinimiser = stein::util::ZeroOrderFunction<FunctionType>;

    
  }
}
