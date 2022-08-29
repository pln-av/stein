#pragma once
#include "types.H"
#include <Eigen/Dense>

namespace stein
{
    namespace target
    {
        template <typename VectorT, typename MatrixT>
        class MVN
        {
            /*
             * Invision that MVN is instantiated with mu == VectorXd, sigsq = MatrixXd, which are
             * both Eigen::Matrix types but note:
             * VectorXd == Eigen::Matrix<double,-1,1>
             * MatrixXd == Eigen::Matrix<double,-1,-1>
             * so we need to different template args.  This is generic enough to handle special
             * matrices, and matrices coming from expression types etc.
             *
             * Last, I will go with row ordering, as I think this will be the most likely access pattern
             * for us where we are just doing repeated matrix multiplications.
             */

        public:
            using Scalar_t = Eigen::MatrixBase<VectorT>::Scalar;
            MVN(const Eigen::MatrixBase<VectorT> &mu, const Eigen::MatrixBase<MatrixT> &sigsq)
                : _mu{mu}, _sigsq{sigsq}, _inv_sigsq{sigsq.inverse()}
            {

                _dimension = static_cast<size_t>(mu.size());
                _norm_const = 1.0 / std::sqrt(std::pow(2.0 * std::numbers::pi_v<Scalar_t>, _dimension) * sigsq.determinant());
            }

            [[nodiscard]] inline Scalar_t operator()(const Eigen::MatrixBase<VectorT> &z) const
            {
                return _norm_const * std::exp(-0.50 * (z - _mu).transpose() * _inv_sigsq * (z - _mu));
            }

            [[nodiscard]] inline VectorT log_grad(const Eigen::MatrixBase<VectorT> &z) const
            {
                return -(z - _mu).transpose() * _inv_sigsq;
            }

            [[nodiscard]] inline size_t dimension(void) const { return _dimension; }

        private:
            // mean and variance.  is the reference ever a problem?
            const Eigen::MatrixBase<VectorT> &_mu;
            const Eigen::MatrixBase<MatrixT> &_sigsq;

            const Matrix_t<Scalar_t> _inv_sigsq;
            Scalar_t _norm_const;
            size_t _dimension;
        };
    }
}