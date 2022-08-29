#pragma once
#include "types.H"
#include <Eigen/Dense>

namespace stein
{
    namespace kernel
    {
        template <typename T>
        class IMQ
        {
            using Scalar_t = T;
        public:
            IMQ(T alpha, T beta) : _alpha(alpha), _beta(beta) {}

            template <typename VectorT>
            [[nodiscard]] inline T operator()(const Eigen::MatrixBase<VectorT> &x, const Eigen::MatrixBase<VectorT> &y) const
            {
                auto &&diff{x - y};
                auto arg = _alpha + diff.squaredNorm();
                return std::pow(arg, _beta);
            }

            template <typename VectorT>
            [[nodiscard]] inline VectorT grad_x(const Eigen::MatrixBase<VectorT> &x, const Eigen::MatrixBase<VectorT> &y) const
            {
                auto &&diff{x - y};
                auto arg = _alpha + diff.squaredNorm();
                return 2.0 * _beta * diff * std::pow(arg, _beta - 1);
            }

            template <typename VectorT>
            [[nodiscard]] inline VectorT grad_y(const Eigen::MatrixBase<VectorT> &x, const Eigen::MatrixBase<VectorT> &y) const
            {
                auto &&diff{x - y};
                auto arg = _alpha + diff.squaredNorm();
                return -2.0 * _beta * diff * std::pow(arg, _beta - 1);
            }

            template <typename VectorT>
            [[nodiscard]] VectorT grad_xy(const Eigen::MatrixBase<VectorT> &x, const Eigen::MatrixBase<VectorT> &y) const
            {
                auto &&diff{x - y};
                auto arg{_alpha + diff.squaredNorm()};
                auto R{std::pow(arg, _beta - 2)};
                auto t1{2.0 * (_beta - 1) * diff.array().abs2() * R};
                auto t2{arg * R};
                return -2.0 * _beta * (t1 + t2);
            }

            template <typename VectorT>
            [[nodiscard]] VectorT grad_yx(const Eigen::MatrixBase<VectorT> &x, const Eigen::MatrixBase<VectorT> &y) const
            {
                auto &&diff{x - y};
                auto arg{_alpha + diff.squaredNorm()};
                auto R{std::pow(arg, _beta - 2)};
                auto t1{2.0 * (_beta - 1) * diff.array().abs2() * R};
                auto t2{arg * R};
                return -2.0 * _beta * (t1 + t2);
            }

        private:
            const T _alpha;
            const T _beta;
        };

    }
}