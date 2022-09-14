//#include <cmath>

#include "mathimf.h"

#include "stein/util/types.H"

namespace stein {
  namespace special {
    template <typename T>
    requires stein::util::is_mkl_float_v<T>
    T gamma(T t)
    {
      if constexpr (stein::util::is_float_v<T>) return tgammaf(t);
      if constexpr (stein::util::is_double_v<T>) return tgamma(t);
    }
    
  }
}
