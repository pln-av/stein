// here is a quick exercise in calling based on types, to
// simplify the mkl calls
#include <iostream>

void algo_f(float i) { std::cout << "algo_f" << std::endl; }
void algo_d(double i) { std::cout << "algo_d" << std::endl; }

// just basic traits for calling mkl functions

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


template <typename T>
void algo(T t)
{
  if constexpr( is_float_v<T> )	algo_f(t);
  else
    if constexpr( is_double_v<T>) algo_d(t);
    else
      static_assert( is_mkl_float_v<T>, "Must be an available mkl float type" );
    
};

int main()

{
  float f = 1.5;
  int i = 1;
  double d = 2.0;
  algo(f);
  algo(d);
  algo(i);  // compile time fail!


   
}

  
