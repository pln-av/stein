#pragma once

#include <Eigen/Dense>

namespace stein
{
    template <typename T = float>
    using Matrix_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    using dMatrix_t = Matrix_t<double>;
    using fMatrix_t = Matrix_t<float>;

    template <typename T = float>
    using Vector_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    using dVector_t = Vector_t<double>;
    using fVector_t = Vector_t<float>;



}